import torch

class Embedding(torch.nn.Module):
    def __init__(self, num_embedding, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embedding = num_embedding
        self.embedding_dim = embedding_dim
        
        self.embedding = torch.nn.Parameter(
            torch.empty((num_embedding, embedding_dim), 
                       device=device, dtype=dtype)
        )
        torch.nn.init.trunc_normal_(self.embedding)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding[token_ids]
