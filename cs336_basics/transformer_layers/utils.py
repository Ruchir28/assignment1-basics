import torch
import math

def softmax(x: torch.Tensor, dim: int):
    max_vals = torch.max(x, dim=dim, keepdim=True).values
    x_shifted = x - max_vals
    exp_x = torch.exp(x_shifted)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None):
    #  q: (B,..,S,d_k)
    #  k: (B,..,S,d_k)
    #  v: (B,..,S,d_v)
    #  mask: (B,..,S,S)

    attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    
    if mask is not None:
        attention_scores = attention_scores.masked_fill(~mask, float("-inf"))

    attention_weights = softmax(attention_scores, dim=-1)

    return torch.matmul(attention_weights, v)

