# dataset.py

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class Vocabulary:

    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4
        for sentence in sentence_list:
            for word in sentence.lower().split(' '):
                frequencies[word] = frequencies.get(word, 0) + 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1


class FlickrDataset(Dataset):

    def __init__(self, root_dir, captions_file, image_list_file, vocab, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.vocab = vocab

        # 1. 读取所有标题
        with open(captions_file, 'r') as f:
            lines = f.readlines()

        self.captions = {}
        for line in lines:
            img_id, caption = line.strip().split('\t')
            img_name = img_id.split('#')[0]
            if img_name not in self.captions:
                self.captions[img_name] = []
            self.captions[img_name].append(caption)

        # 2. 读取指定的图片列表 (训练/测试/验证)
        with open(image_list_file, 'r') as f:
            self.image_list = [line.strip() for line in f.readlines()]

        # 3. 扁平化数据
        self.flat_data = []
        for img_name in self.image_list:
            for caption in self.captions[img_name]:
                self.flat_data.append((img_name, caption))

    def __len__(self):
        return len(self.flat_data)

    def __getitem__(self, index):
        img_name, caption_text = self.flat_data[index]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # 将标题文本数值化
        numericalized_caption = [self.vocab.stoi["<START>"]]
        numericalized_caption += [self.vocab.stoi.get(word, self.vocab.stoi["<UNK>"]) for word in
                                  caption_text.lower().split(' ')]
        numericalized_caption.append(self.vocab.stoi["<END>"])

        return image, torch.tensor(numericalized_caption)


class Collate:
    """用于DataLoader的辅助类，处理变长序列的填充"""

    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)
        return imgs, targets