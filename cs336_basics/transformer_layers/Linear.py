import torch
import torch.nn.init as init

class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, 
                 bias: bool = False, device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), 
                       device=device, dtype=dtype)
        )
        init.trunc_normal_(self.weight, mean=0.0, std=0.02)
        
        if bias:
            self.bias = torch.nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype)
            )
            init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor):
        return torch.nn.functional.linear(x, self.weight, self.bias)
