import os
import glob

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


# ========== 模型定义（比训练代码多了一个predict） ==========
class RnnClsNet(nn.Module):
    def __init__(
        self, vocab_size=5, embed_dim=5, hidden_dim=16, num_layers=2, cls_num=5
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=-1)
        self.rnn = nn.RNN(
            embed_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, cls_num)

    def forward(self, x, seq_lengths):
        embedded = self.embedding(x)
        pack = pack_padded_sequence(
            embedded, seq_lengths, enforce_sorted=False, batch_first=True
        )
        out, h_n = self.rnn(pack)
        out = self.fc(h_n[-1])
        return out

    def predict(self, x):
        """
        输入一个List序列或者torch.Tensor序列，返回类别号
        x: List或者torch.Tensor
        返回: cls_num
        """
        if isinstance(x, list):
            x = torch.tensor(x, dtype=torch.long).unsqueeze(0)  # 添加 batch 维度
        elif isinstance(x, torch.Tensor):
            if x.dim() == 1:
                x = x.unsqueeze(0)  # 添加 batch 维度
        else:
            raise TypeError("Input x must be a list or torch.Tensor")

        seq_lengths = torch.tensor([x.size(1)] * x.size(0), dtype=torch.long)

        with torch.no_grad():
            output = self.forward(x, seq_lengths)
            output = nn.functional.softmax(output, dim=1)
            pred = torch.argmax(output, dim=1)

        return pred.item() if pred.size(0) == 1 else pred


# ========== 主程序：加载模型 + 终端交互预测 ==========
if __name__ == "__main__":
    # ===== 配置超参数 =====
    save_dir = "./output"

    # ===== 模型初始化 =====
    args = torch.load(os.path.join(save_dir, "args.pt"), weights_only=False)
    model = RnnClsNet(
        vocab_size=args.max_number
        + 1,  # 输出和数字对齐，注意数组包含0，有max_number+1个输入token，下同
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        cls_num=args.max_number + 1,
    )
    model.eval()  # 设置为评估模式

    # 读取最新的chckpoints
    lastest_checkpoints = max(
        glob.glob(os.path.join(save_dir, "model_step_*.pt")),
        key=lambda x: int(x.split("_")[-1].split(".")[0]),
    )
    print(f"使用模型权重：{lastest_checkpoints}")
    model.load_state_dict(
        torch.load(lastest_checkpoints, map_location=torch.device("cpu"))
    )  # 加载权重

    # ===== 交互式预测 =====
    print(
        f"\n请输入数字序列（空格分隔，如：1 2 3），注意输入数字要在0-{args.max_number}之间，输入 'quit' 退出："
    )
    while True:
        user_input = input(">>> ").strip()
        if user_input.lower() == "quit":
            print("👋 退出程序。")
            break

        try:
            # 解析输入为整数列表
            seq = list(map(int, user_input.split()))
            if len(seq) == 0:
                print("⚠️  输入为空，请重新输入。")
                continue
            # 预测
            pred_class = model.predict(seq)
            print(f"⌨️ 输入序列: {seq}")
            print(f"🎯 模型预测结果: {pred_class}")
            print(
                f"✅ 输入序列的求和除{args.max_number+1}余数为: {sum(seq)%(args.max_number+1)}"
            )

        except ValueError:
            print("⚠️  输入格式错误，请输入空格分隔的整数，如：1 2 3")
        except Exception as e:
            print(f"❌ 预测出错: {e}")
