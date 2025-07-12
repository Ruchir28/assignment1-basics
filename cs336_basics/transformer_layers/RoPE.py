import torch
import torch.nn as nn
class RoPE(nn.Module):
    def __init__(self,theta: float,d_k:int,max_seq_len:int, device:torch.device = None):
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.theta = theta
    
        
        inv_frequencies = 1.0 / (self.theta ** (torch.arange(0,self.d_k,2) / self.d_k)) #(d_k / 2)
        
        t = torch.arange(max_seq_len, device=device).float() #(max_seq_len)
        
        freqs = torch.outer(t,inv_frequencies) # (max_seq_len, dk // 2)
        
        self.register_buffer('cos_cached', torch.cos(freqs)) # (max_seq_len, dk // 2)
        
        self.register_buffer('sin_cached', torch.sin(freqs)) # (max_seq_len, dk // 2)
        
    def forward(self,x: torch.Tensor, positions: torch.Tensor | None = None):
        # x: [..., seq_len,d_k]
        # positions: [..., seq_len]
        
        seq_len = x.shape[-2]
        
        if positions == None:
            positions = torch.arange(seq_len, device=x.device)
        
        cos = self.cos_cached[positions]
        
        sin = self.sin_cached[positions]    
        
        x_even = x[...,::2] 
        
        x_odd = x[...,1::2]
        
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_odd * cos + x_even * sin
        
        x_rotated = torch.empty_like(x)
        
        x_rotated[...,::2]  = x_rotated_even
        
        x_rotated[...,1::2] = x_rotated_odd
        
        return x_rotated
        
        
        
        
        
            