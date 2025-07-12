import torch
import torch.nn as nn

class SwigluFFN(nn.Module):
    def __init__(self,d_model: int,d_ff: int | None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff 
        
        if self.d_ff == None:
            self.d_ff = (8 * d_model) // 3
        
        self.gate = nn.Linear(d_model,self.d_ff, bias=False)
        
        self.signal = nn.Linear(d_model,self.d_ff, bias=False)
        
        self.down_proj = nn.Linear(self.d_ff, d_model, bias=False)
        
    def forward(self,x: torch.Tensor):
        # (B,S,d_model)
        
        gate = self.gate(x)
        
        silu = gate * torch.sigmoid(gate)
        
        signal = self.signal(x)
        
        swiglu = silu * signal
        
        return self.down_proj(swiglu)
        