import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self,d_model: int,eps: float = 1e-5,device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self,x: torch.Tensor):
        # x: (B,S,d_model)
        
        in_dtype = x.dtype
        
        x = x.to(torch.float32) ## to prevent overflow upcast to float32
        
        rms_a = torch.sqrt(((torch.mean(x*x,dim=-1,keepdim=True))+ self.eps)) # (B,S,1)
        
        answer = (x/rms_a) * self.weight
        
        answer = answer.to(in_dtype)
        
        return answer