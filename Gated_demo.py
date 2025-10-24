from modulefinder import Module

import torch
import torch.nn as nn

torch.manual_seed(42)

class GatedCrossAttention(nn.Module):
    def __init__(self,model_dim=128,num_heads=4):
        # modle_dim 特征向量的维度
        # num_heads 多头注意力的头数
        super().__init__()

        #交叉注意力层次
        self.cross_attention=nn.MultiheadAttention(
            num_heads=num_heads,
            embed_dim=model_dim,
            batch_first=True, #让输入数据的batch维度排在最前面
        )

        #门控机制，初始化为0，控制信息流量的阀门
        self.gate=nn.Parameter(torch.tensor(0.0))

        #归一化
        self.layer_norm=nn.LayerNorm(model_dim)

    def forward(self,text_features,vision_features):
        #输入模拟的文字和视觉特征
        attn_output,_=self.cross_attention(
            query=text_features,
            key=vision_features,
            value=vision_features,
        )

        #应用门控
        gated_attn_output=torch.tanh(self.gate)*attn_output

        #融合特征
        updated_text_features=self.layer_norm(gated_attn_output)

        return updated_text_features

BATCH_SIZE=2
TEXT_SEQ_LEN=10
VISION_SEQ_LEN=64
MODEL_DIM=128

text_features = torch.randn(BATCH_SIZE, TEXT_SEQ_LEN, MODEL_DIM)
vision_features = torch.randn(BATCH_SIZE, VISION_SEQ_LEN, MODEL_DIM)

gated_attention_layer=GatedCrossAttention(model_dim=MODEL_DIM)

print("="*30)
print("▶ 阶段一：训练开始前 (门是关闭的)")
print("="*30)

# 获取初始的门控参数值
initial_gate_value = gated_attention_layer.gate.item()
print(f"初始门控 (Gate) 参数值: {initial_gate_value:.4f}")

output_before_training = gated_attention_layer(text_features, vision_features)

# 比较输出和输入的差异
diff_before = torch.mean((output_before_training - text_features)**2)
print(f"输出与原始文本输入的均方差: {diff_before:.6f}")

# 假设在真实训练中，优化器发现需要融合视觉信息，
# 于是它通过反向传播更新了门控参数 `gate` 的值。
# 我们在这里手动模拟这个更新过程。
print("\n... (模拟训练中) ...\n")
with torch.no_grad(): # 在非训练模式下修改参数
    # 我们手动将 gate 的值从 0 调大
    gated_attention_layer.gate.data += 1.5

print("="*30)
print(" 阶段二：训练一段时间后 (门被打开)")
print("="*30)

# 获取更新后的门控参数值
final_gate_value = gated_attention_layer.gate.item()
print(f"更新后门控 (Gate) 参数值: {final_gate_value:.4f}")

# 使用完全相同的输入，再次运行前向传播
output_after_training = gated_attention_layer(text_features, vision_features)

# 再次比较输出和输入的差异
diff_after = torch.mean((output_after_training - text_features)**2)
print(f"输出与原始文本输入的均方差: {diff_after:.6f}")