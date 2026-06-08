from torch import nn
import torchinfo
from transformer import Encoder, MultiheadAttention
from typing import Optional


import torch


class MultiLayerPerceptron(nn.Module):
    def __init__(self, embedding_dim: int=768, feedforward_dim: int=3072, dropout: float=0.1):
        super().__init__()
        self.seq_layer = nn.Sequential(nn.Linear(in_features=embedding_dim, out_features=feedforward_dim),
                                       nn.GELU(),
                                       nn.Dropout(dropout),
                                       nn.Linear(in_features=feedforward_dim, out_features=embedding_dim),
                                       nn.Dropout(dropout))
    
    def forward(self, X: torch.Tensor):
        return self.seq_layer(X)


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim: int=768, num_heads: int=12, feedforward_dim: int=3072, mlp_dropout: float=0.1, attn_dropout: float=0.0):
        super().__init__()
        self.mha = MultiheadAttention(embedding_dim, num_heads, embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(attn_dropout)
        self.norm_1 = nn.LayerNorm(embedding_dim)
        self.ff = MultiLayerPerceptron(embedding_dim, feedforward_dim, mlp_dropout)
        self.norm_2 = nn.LayerNorm(embedding_dim)

    def forward(self, source: torch.Tensor, key_padding_mask: Optional[torch.Tensor]=None, attn_mask: Optional[torch.Tensor]=None):
        # encoder sometimes uses attn_mask to block token-to-token attention
        # though attn_mask is more often used as a causal mask in decoder
        normalized_source = self.norm_1(source)
        mha_output = self.mha(normalized_source, normalized_source, normalized_source, key_padding_mask=key_padding_mask, attn_mask=attn_mask)[0]
        mha_output = self.norm_1(source + self.dropout(mha_output))
        return mha_output + self.ff(self.norm_2(mha_output))


class VisionTransformer(nn.Module):
    def __init__(self,
                 height: int,
                 width: int,
                 class_num: int,
                 d_model: int=768,
                 feedforward_dim: int=3072,
                 patch_size: int=16,
                 num_heads: int=12,
                 num_layers: int=12,
                 embedding_dropout: float=0.1,
                 mlp_dropout: float=0.1,
                 attn_dropout: float=0.1):

        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError("Input image is not divisible to the patch")
    
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=3, out_channels=d_model, kernel_size=patch_size, stride=patch_size)
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

        class_token = torch.zeros((1, d_model))
        nn.init.trunc_normal_(class_token, std=0.2)
        self.class_token = nn.Parameter(class_token, True)

        positional_embedding = torch.zeros((height * width // patch_size ** 2 + 1, d_model))
        nn.init.trunc_normal_(positional_embedding, std=0.2)
        self.positional_embedding = nn.Parameter(positional_embedding, True)

        self.dropout = nn.Dropout(embedding_dropout)
        encoder_layer = EncoderLayer(embedding_dim=d_model, num_heads=num_heads, feedforward_dim=feedforward_dim, mlp_dropout=mlp_dropout, attn_dropout=attn_dropout)
        self.encoder = Encoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(in_features=d_model, out_features=class_num))
    
    def forward(self, image: torch.Tensor):
        # turn an image into patch embeddings and flatten 2d embeddings to 1d
        patch_embedding = self.flatten(self.conv2d(image))   # (batch, d_model, patch_num)
        patch_embedding = patch_embedding.transpose(1, 2)

        # concatenate class token to patch embeddings and add positional embeddings
        batch_size = image.shape[0]
        source = torch.cat([self.class_token.expand(batch_size, -1, -1), patch_embedding], dim=1) + self.positional_embedding   # (batch, patch_num + 1, d_model)
        logits = self.encoder(self.dropout(source))

        return self.head(logits[:, 0])


def main():
    height = width = 224
    model = VisionTransformer(height=height, width=width, class_num=10)
    torchinfo.summary(model, input_size=(32, 3, height, width))

if __name__ == "__main__":
    main()