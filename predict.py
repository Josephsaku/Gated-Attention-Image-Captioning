# predict.py

import torch
from torchvision import transforms
from PIL import Image

# 确保导入与训练时完全相同的模型和词汇表类定义
from model import SimpleVisionLanguageModel
from dataset import Vocabulary


# def generate_caption(model, image_path, vocab, transform, device, max_length=50):
#
#     model.eval()
#     result_caption = [vocab.stoi["<START>"]]
#
#     try:
#         img = Image.open(image_path).convert("RGB")
#         image_tensor = transform(img).unsqueeze(0).to(device)
#     except Exception as e:
#         return f"无法加载图片: {e}"
#
#     for _ in range(max_length):
#         text_tensor = torch.LongTensor(result_caption).unsqueeze(0).to(device)
#         with torch.no_grad():
#             logits, gate_value = model(image_tensor, text_tensor)
#
#         predicted_idx = logits.argmax(2)[0, -1].item()
#         result_caption.append(predicted_idx)
#
#         if predicted_idx == vocab.stoi["<END>"]:
#             break
#
#     generated_words = [vocab.itos[idx] for idx in result_caption]
#     final_caption = ' '.join(generated_words[1:-1])
#
#     # 打印一下推理时的gate值
#     print(f"推理时，门控 (Gate) 参数值: {gate_value:.4f}")
#
#     return final_caption


def generate_caption(model, image_path, vocab, transform, device, max_length=50, top_k=10):
    """使用加载的模型为本地图片生成标题 (使用Top-K采样策略)"""
    model.eval()
    result_caption = [vocab.stoi["<START>"]]

    try:
        img = Image.open(image_path).convert("RGB")
        image_tensor = transform(img).unsqueeze(0).to(device)
    except Exception as e:
        return f"无法加载图片: {e}"

    for _ in range(max_length):
        text_tensor = torch.LongTensor(result_caption).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, gate_value = model(image_tensor, text_tensor)

        # --- 核心修改：从 argmax 变为 Top-K 采样 ---
        # 1. 只看最后一个词的预测
        last_word_logits = logits[0, -1, :]

        # 2. 找到概率最高的 K 个词的 logits，并把其他词的概率设为负无穷
        #    这样可以确保我们只在这 K 个词里选择
        top_k_vals, top_k_indices = torch.topk(last_word_logits, top_k)

        # 创建一个全为负无穷的张量
        filtered_logits = torch.full_like(last_word_logits, float('-inf'))
        # 只保留 top_k 词的原始 logits
        filtered_logits.scatter_(0, top_k_indices, top_k_vals)

        # 3. 将这些 logits 转换为概率分布
        probabilities = torch.nn.functional.softmax(filtered_logits, dim=-1)

        # 4. 从整个词汇表的概率分布中随机抽取一个词
        predicted_idx = torch.multinomial(probabilities, 1).item()
        # --- 修改结束 ---

        result_caption.append(predicted_idx)

        if predicted_idx == vocab.stoi["<END>"]:
            break

    generated_words = [vocab.itos[idx] for idx in result_caption]
    final_caption = ' '.join(generated_words[1:-1])

    print(f"推理时，门控 (Gate) 参数值: {gate_value:.4f}")

    return final_caption


if __name__ == "__main__":
    # --- 1. 设置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "gated_attn_model.pth"
    image_path = "data/cat_test.jpg"  # 确保你已经把猫的图片下载到了这个路径

    print(f"将使用设备: {device}")
    print(f"正在从 {model_path} 加载已训练的模型...")

    # --- 2. 加载模型和词汇表 ---
    checkpoint = torch.load(model_path, map_location=device)
    vocab = checkpoint['vocab']

    # --- 3. 重建模型结构并加载学习到的权重 ---
    model = SimpleVisionLanguageModel(vocab_size=len(vocab)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print("模型加载成功！")

    # --- 4. 定义图像预处理 (必须与训练时完全一致) ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- 5. 生成标题 ---
    print(f"\n正在为本地图片生成标题: {image_path}")
    caption = generate_caption(model, image_path, vocab, transform, device)
    print("\n" + "=" * 30)
    print(f"模型生成的标题是: '{caption}'")
    print("=" * 30)