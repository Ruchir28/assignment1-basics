import torch
from cs336_basics.transformer_layers.CausalMultiHeadAttention import CausalMultiHeadAttention
from cs336_basics.transformer_layers.RmsNorm import RMSNorm
from cs336_basics.transformer_layers.SwigluFFN import SwigluFFN

class TransFormer(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super().__init__()
        self.pre_rms_norm = RMSNorm(d_model=d_model)
        self.causal_multi_head_attention = CausalMultiHeadAttention(d_model, num_heads, rope_enabled=True, theta=theta, max_seq_len=max_seq_len)
        self.swiglu_ffn = SwigluFFN(d_model, d_ff)
        self.post_rms_norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        # Pre-norm transformer: LayerNorm -> Attention -> Add residual
        normed_x = self.pre_rms_norm(x)
        
        # Create causal mask
        B, S, d_model = x.shape
        mask = torch.tril(torch.ones(S, S, device=x.device), diagonal=0).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)
        
        causal_multi_head_attention_output = self.causal_multi_head_attention(normed_x, mask, token_positions)
        x = x + causal_multi_head_attention_output  # Add residual connection
        
        # Pre-norm transformer: LayerNorm -> FFN -> Add residual  
        normed_x = self.post_rms_norm(x)
        ffn_output = self.swiglu_ffn(normed_x)
        
        return x + ffn_output  # Add residual connection
