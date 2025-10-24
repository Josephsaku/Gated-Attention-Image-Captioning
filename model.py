# model.py

import torch
import torch.nn as nn
import torchvision.models as models


class GatedCrossAttention(nn.Module):

    def __init__(self, model_dim=128, num_heads=4):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.gate = nn.Parameter(torch.tensor(0.0))
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, text_features, vision_features):
        attn_output, _ = self.cross_attention(
            query=text_features,
            key=vision_features,
            value=vision_features
        )
        gated_attn_output = torch.tanh(self.gate) * attn_output
        updated_text_features = self.layer_norm(text_features + gated_attn_output)
        return updated_text_features


class SimpleVisionLanguageModel(nn.Module):

    def __init__(self, vocab_size, embed_dim=256, vision_dim=512):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.vision_encoder = nn.Sequential(*list(resnet.children())[:-1])
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        self.text_embedding = nn.Embedding(vocab_size, embed_dim)
        self.vision_projection = nn.Linear(vision_dim, embed_dim)
        self.fusion_layer = GatedCrossAttention(model_dim=embed_dim)
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, image_tensor, text_sequence):
        self.vision_encoder.eval()
        with torch.no_grad():
            image_features_raw = self.vision_encoder(image_tensor).flatten(1)
        image_features_proj = self.vision_projection(image_features_raw)
        image_features = image_features_proj.unsqueeze(1)

        text_features = self.text_embedding(text_sequence)
        fused_features = self.fusion_layer(text_features, image_features)
        logits = self.output_layer(fused_features)

        return logits, self.fusion_layer.gate.item()