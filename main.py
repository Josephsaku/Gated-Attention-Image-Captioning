# main.py (修改后)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import Vocabulary, FlickrDataset, Collate
from model import SimpleVisionLanguageModel


def train():
    # --- 1. 设置和准备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"将使用设备: {device}\n")

    image_folder = "data/Flicker8k_Dataset"
    captions_file = "data/Flickr8k.token.txt"
    train_images_file = "data/Flickr_8k.trainImages.txt"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- 2. 构建词汇表和数据集 ---
    print("正在构建词汇表...")
    with open(captions_file, 'r', encoding='utf-8') as f:  # 增加 encoding='utf-8' 更健壮
        all_captions_text = [line.strip().split('\t')[1] for line in f.readlines()]
    vocab = Vocabulary(freq_threshold=5)
    vocab.build_vocabulary(all_captions_text)
    print(f"词汇表构建完毕，大小: {len(vocab)}")

    print("正在创建数据集和DataLoader...")
    train_dataset = FlickrDataset(
        root_dir=image_folder,
        captions_file=captions_file,
        image_list_file=train_images_file,
        vocab=vocab,
        transform=transform
    )
    pad_idx = vocab.stoi["<PAD>"]
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=Collate(pad_idx=pad_idx),
        num_workers=2  # 可以尝试增加 num_workers 来加速数据加载
    )

    # --- 3. 初始化模型、损失函数和优化器 ---
    torch.manual_seed(42)
    model = SimpleVisionLanguageModel(vocab_size=len(vocab)).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 50

    # --- 4. 训练循环 ---
    print("\n开始训练...")
    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for images, captions in progress_bar:
            images, captions = images.to(device), captions.to(device)
            input_captions, target_captions = captions[:, :-1], captions[:, 1:]

            outputs, _ = model(images, input_captions)
            loss = criterion(outputs.reshape(-1, len(vocab)), target_captions.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=loss.item())

    print("训练完毕!")

    # --- 5. 保存模型和词汇表 ---
    # 【核心修改】
    save_path = "gated_attn_model.pth"
    print(f"正在保存模型至 {save_path}...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
    }, save_path)
    print("模型保存成功！")


if __name__ == "__main__":
    train()