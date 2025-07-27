import torch

from cs336_basics.transformer_layers.Embedding import Embedding
from cs336_basics.transformer_layers.Transformer import TransFormer
from cs336_basics.transformer_layers.RmsNorm import RMSNorm

class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size,d_model)
        self.blocks = torch.nn.ModuleList([
            TransFormer(d_model,num_heads,d_ff,context_length,rope_theta)
            for _ in range(num_layers)
        ])
        
        self.ln_final = RMSNorm(d_model)
        
        self.lm_head = torch.nn.Linear(d_model,vocab_size,bias=False)
        
    def forward(self,in_indices: torch.Tensor):
        
        B, S = in_indices.shape
        
        x = self.token_embeddings(in_indices) # (B,S,d_model)
        
        pos = torch.arange(S,device=x.device).unsqueeze(0).expand(B,-1) # (B,S)
        
        for block in self.blocks:
            x = block(x,pos)
        
        x = self.ln_final(x)
        
        logits = self.lm_head(x)
        
        return logits
            
        
        
        