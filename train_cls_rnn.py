import os
import argparse
import random
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import swanlab


# ========== 1. 数据部分 ==========


# 可变长数据集，X为一个0-N的序列，Y为该序列的求和除N+1的余数，范围也是0-N
class SeqSumModDataset(IterableDataset):
    def __init__(self, total_samples=None, min_seq_len=2, max_seq_len=5, max_number=4):
        """
        :param total_samples: 总样本数（None = 无限生成）
        :param min_len: 最短字母序列长度
        :param max_len: 最长字母序列长度
        :param max_number: 序列会出现的最大数字
        """
        self.total_samples = total_samples
        self.min_len = min_seq_len
        self.max_len = max_seq_len
        self.max_number = max_number
        self.count = 0  # 用于限制样本总数

    def __iter__(self):
        self.count = 0  # 每次重新迭代时重置计数器
        return self

    def __next__(self):
        # 用于控制epoch
        if self.total_samples is not None and self.count >= self.total_samples:
            raise StopIteration

        # 动态生成一个样本
        seq_length = random.randint(self.min_len, self.max_len)
        input_seq = [random.randint(0, self.max_number) for _ in range(seq_length)]
        target_num = sum(input_seq) % (self.max_number + 1)
        self.count += 1
        return input_seq, seq_length, target_num


# ========== 2. 模型部分（示例：字符级 简单的RNN模型） ==========


class RnnClsNet(nn.Module):
    def __init__(
        self, vocab_size=5, embed_dim=5, hidden_dim=16, num_layers=2, cls_num=5
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size + 1, embed_dim, padding_idx=-1
        )  # 多一个填充位
        self.rnn = nn.RNN(
            embed_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, cls_num)

    def forward(self, x, seq_lengths):
        # x: [batch_size, seq_len] （字符索引）
        embedded = self.embedding(x)
        pack = pack_padded_sequence(
            embedded, seq_lengths, enforce_sorted=False, batch_first=True
        )
        out, h_n = self.rnn(pack)
        out = self.fc(h_n[-1])
        return out


# ========== 3. 训练脚本 ==========


def train(args, device):
    # 初始化SwanLab记录日志
    swanlab.init(experiment_name=args.run_name, config=args)

    # 创建数据集
    dataset = SeqSumModDataset(  # 训练集，无限个
        total_samples=None,
        min_seq_len=args.min_seq_len,
        max_seq_len=args.max_seq_len,
        max_number=args.max_number,
    )

    def pad_and_tensor(data, padding_value=5):
        """填充函数"""
        input_seqs, seq_lengths, target_nums = zip(*data)
        input_seqs = [torch.LongTensor(input_seq) for input_seq in input_seqs]
        seq_lengths = torch.LongTensor(seq_lengths)
        target_nums = torch.LongTensor(target_nums)
        input_seqs = pad_sequence(
            input_seqs, batch_first=True, padding_value=padding_value
        )
        return input_seqs, seq_lengths, target_nums

    pad_and_tensor = partial(pad_and_tensor, padding_value=args.max_number + 1)
    train_dataloader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=pad_and_tensor
    )
    eval_dataset = SeqSumModDataset(  # 测试集，一共100个（注意测试集样本并非固定）
        total_samples=100,
        min_seq_len=args.min_seq_len,
        max_seq_len=args.max_seq_len,
        max_number=args.max_number,
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=args.batch_size, collate_fn=pad_and_tensor
    )

    # 创建模型
    model = RnnClsNet(
        vocab_size=args.max_number
        + 1,  # 输出和数字对齐，注意数组包含0，有max_number+1个输入token，下同
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        cls_num=args.max_number + 1,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    swanlab.log({"Model Params": total_params})
    model.train()

    # 准备优化器和损失函数
    criterion = nn.CrossEntropyLoss().to(device)  # 示例用回归损失
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 余弦学习率衰减
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.total_steps, eta_min=args.min_lr
    )

    if args.warmup_step > 0:
        # 组合调度器：先 warmup，再 cosine
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,  # 起始缩放比例
            end_factor=1.0,  # 结束时为 100%
            total_iters=args.warmup_step,
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, scheduler],
            milestones=[args.warmup_step],  # 在第 warmup_epochs 个 epoch 后切换调度器
        )

    # 开始训练
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(args, os.path.join(args.save_dir, "args.pt"))  # 保存模型参数
    step = 0
    data_iter = iter(train_dataloader)
    while step < args.total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            # 如果是有限数据集，重新创建迭代器
            data_iter = iter(train_dataloader)
            batch = next(data_iter)

        input_seqs, seq_lengths, target_nums = batch
        input_seqs = input_seqs.to(device)
        target_nums = target_nums.to(device)
        # ========== 前向 + 损失 + 反向 ==========
        optimizer.zero_grad()
        outputs = model(input_seqs, seq_lengths)
        print
        loss = criterion(outputs, target_nums)
        loss.backward()
        optimizer.step()
        scheduler.step()
        step += 1

        # ========== 评估 ==========
        if step % args.eval_every == 0:
            accuracy = eval_loop(model, eval_dataloader, device)
            print(f"##### Eval [{step}/{args.total_steps}], Acc: {accuracy:.4f}")
            swanlab.log({"accuracy": accuracy}, step=step)

        # ========== 日志 ==========
        if step % args.log_every == 0:
            print(f"Step [{step}/{args.total_steps}], Loss: {loss.item():.4f}")
            current_lr = scheduler.get_last_lr()[0]
            swanlab.log({"loss": loss.item(), "current_lr": current_lr}, step=step)
        # ========== 保存模型 ==========
        if args.save_every and step % args.save_every == 0:
            torch.save(
                model.state_dict(), os.path.join(args.save_dir, f"model_step_{step}.pt")
            )
            path = os.path.join(args.save_dir, f"model_step_{step}.pt")
            print(f"模型已保存: {path}")

    torch.save(model.state_dict(), os.path.join(args.save_dir, f"model_step_{step}.pt"))
    path = os.path.join(args.save_dir, f"model_step_{step}.pt")
    print(f"模型已保存: {path}")
    print("训练完成！")


# ========== 4. 评估程序 ==========


def eval_loop(model, dataloader, device):
    model.eval()  # 切换到评估模式（关闭 dropout、BN 等）
    total_correct = 0
    total_samples = 0

    with torch.no_grad():  # 整个循环放在 no_grad 里更高效
        for batch in dataloader:
            input_seqs, seq_lengths, target_nums = batch

            input_seqs = input_seqs.to(device)
            target_nums = target_nums.to(device)

            # 模型推理
            outputs = model(input_seqs, seq_lengths)
            # 获取预测类别
            pred = torch.argmax(outputs, dim=1)  # shape: (batch_size,)
            # 累计正确预测数和样本总数
            total_correct += (pred == target_nums).sum().item()
            total_samples += target_nums.size(0)

    model.train()  # 切换回训练模式
    # 计算整体准确率
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    return accuracy


# ========== 5. 启动程序 ==========

if __name__ == "__main__":
    # 使用设备
    device = "cpu"  # for debug
    # device = "mps"  # for mac（我的M2电脑发现，用这个比cpu模式还慢😭）
    # device = "npu"  # for ascend

    # 超参数
    parser = argparse.ArgumentParser(description="Training Configuration")
    # 数据集参数
    parser.add_argument("--min_seq_len", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=5)
    parser.add_argument("--max_number", type=int, default=9)
    # 模型参数
    parser.add_argument("--embed_dim", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=1)
    # 训练参数
    parser.add_argument("--total_steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--min_lr", type=float, default=0.0001)
    parser.add_argument("--warmup_step", type=float, default=100)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--save_dir", type=str, default="./output/")
    parser.add_argument("--run_name", type=str, default="baseline-lr1e2")
    args = parser.parse_args()
    # 开始训练
    train(args, device)
