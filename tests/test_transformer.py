from torch import nn
from src.transformer import MultiheadAttention

import torch
import torch.testing as tt


def test_multihead_attention():
    base = nn.MultiheadAttention(embed_dim=512, num_heads=8, bias=False, batch_first=True)
    custom = MultiheadAttention()
    with torch.no_grad():
        q_weight, k_weight, v_weight = base.in_proj_weight.chunk(3, dim=0)
        custom.W_Q.copy_(q_weight.T)
        custom.W_K.copy_(k_weight.T)
        custom.W_V.copy_(v_weight.T)
        custom.W_O.copy_(base.out_proj.weight.T)

    x = torch.rand((1, 10, 512))
    key_padding_mask = torch.rand((1, 10)) > 0.5
    attn_mask = torch.rand((10, 10)) > 0.5
    actual = custom(x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
    expected = base(x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)[0]
    tt.assert_close(actual, expected, atol=1e-6, rtol=1e-5)