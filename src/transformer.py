from typing import Optional
from torch import nn

import math
import torch

class MultiheadAttention(nn.Module):
    def __init__(self, embedding_dim: int = 512, num_heads: int = 8, k_dim: int = 512, v_dim: int = 512):
        # default values are the hyperparameters the paper used
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("Embedding dimension is not divisible to number of heads")
        
        self.num_heads = num_heads

        # parameters from multiple heads are stacked as [W_1, ..., W_h]
        # W_n is of size (embedding_dim, embedding_dim // head)
        self.W_Q = nn.Parameter(torch.empty(size=(embedding_dim, embedding_dim)))
        self.W_K = nn.Parameter(torch.empty(size=(k_dim, embedding_dim)))
        self.W_V = nn.Parameter(torch.empty(size=(v_dim, embedding_dim)))
        self.W_O = nn.Parameter(torch.empty((embedding_dim, embedding_dim)))
        nn.init.xavier_uniform_(self.W_Q)
        nn.init.xavier_uniform_(self.W_K)
        nn.init.xavier_uniform_(self.W_V)
        nn.init.xavier_uniform_(self.W_O)
    
    def forward(self,
                query: torch.Tensor,   # (batch, target_length, embedding_dim)
                key: torch.Tensor,     # (batch, source_length, k_dim)
                value: torch.Tensor,   # (batch, source_length, v_dim)
                key_padding_mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None):
        # the key_padding_mask is used to mask out the paddings in sequences
        # it should have shape (N, S) where N is the size of batches and S is the max sequence length
        # attn_mask is for the masked multi-head attention in the decoder, masking out the leftward relationship of tokens
        batch, target_length, embedding_dim = query.shape
        source_length = key.shape[1]

        if value.shape[1] != source_length:
            raise ValueError("key and value must have the same length")
    
        d_k = embedding_dim // self.num_heads
        Q = (query @ self.W_Q).view(size=(batch, target_length, self.num_heads, d_k)).transpose(1, 2)
        K = (key @ self.W_K).view(size=(batch, source_length, self.num_heads, d_k)).transpose(1, 2)
        V = (value @ self.W_V).view(size=(batch, source_length, self.num_heads, d_k)).transpose(1, 2)

        score = Q @ K.transpose(-1, -2) / math.sqrt(d_k)

        # assuming both are boolean masks
        # True means ignore that position
        if key_padding_mask is not None:
            score.masked_fill_(key_padding_mask[:, torch.newaxis, torch.newaxis, :] == 1, float("-inf"))
        
        if attn_mask is not None:
            score.masked_fill_(attn_mask == 1, float("-inf"))

        weights = torch.softmax(score, dim=-1)
        
        return torch.cat((weights @ V).unbind(dim=1), dim=-1) @ self.W_O, weights
