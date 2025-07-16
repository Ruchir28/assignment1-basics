import torch
import math

from cs336_basics.transformer_layers.utils import softmax
from cs336_basics.transformer_layers.RoPE import RoPE


class CausalMultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope_enabled: bool = False, theta: float = 10000, max_seq_len: int = 1024):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        self.q_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self.k_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self.v_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self.o_proj = torch.nn.Linear(d_model, d_model, bias=False)

        self.rope_enabled = rope_enabled
        if self.rope_enabled:
            self.rope = RoPE(theta=theta,d_k=self.d_k,max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, token_positions: torch.Tensor | None = None):
        # x: (B, S, d_model)
        # mask: (B, S, S) or (B, 1, S, S) or (B, num_heads, S, S)

        B, S, d_model = x.shape

        assert d_model == self.d_model, "Input dimension must match model dimension"

        q = self.q_proj(x)  # (B, S, d_model)
        k = self.k_proj(x)  # (B, S, d_model)
        v = self.v_proj(x)  # (B, S, d_model)

        q = q.view(B, S, self.num_heads, self.d_k).transpose(1, 2)  # (B, num_heads, S, d_k)
        k = k.view(B, S, self.num_heads, self.d_k).transpose(1, 2)  # (B, num_heads, S, d_k)
        v = v.view(B, S, self.num_heads, self.d_v).transpose(1, 2)  # (B, num_heads, S, d_v)

        if self.rope_enabled:
            q = self.rope(q,token_positions)
            k = self.rope(k,token_positions)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, num_heads, S, S)

        if mask is not None:
            # Handle different mask shapes
            if mask.dim() == 3:  # (B, S, S)
                mask = mask.unsqueeze(1)  # (B, 1, S, S)
            elif mask.dim() == 4 and mask.shape[1] == 1:  # (B, 1, S, S)
                pass  # Already correct shape
            elif mask.dim() == 4 and mask.shape[1] == self.num_heads:  # (B, num_heads, S, S)
                pass  # Already correct shape
            else:
                raise ValueError(f"Unexpected mask shape: {mask.shape}")
            
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        attn_weights = softmax(attn_scores, dim=-1)  # (B, num_heads, S, S)

        attn_output = torch.matmul(attn_weights, v)  # (B, num_heads, S, d_v)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, d_model)  # (B, S, d_model)
        
        output = self.o_proj(attn_output)  # (B, S, d_model)
        
        return output
            
        
  