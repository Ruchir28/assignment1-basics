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

def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor):
    # inputs : (batch_size, vocab_size)
    # targets : (batch_size)

    max_values = torch.max(inputs,dim=-1,keepdim=True).values #(batch_size, 1)

    normalized_inputs = inputs - max_values
    
    exp_x = torch.exp(normalized_inputs) # (batch_size,vocab_size)
    
    exp_x_sum = torch.sum(exp_x,dim=-1) #(batch_size)
    
    predicted_prob = torch.gather(normalized_inputs,dim=-1,index=targets.unsqueeze(-1)).squeeze(1) #(batch_size)

    # cross entrop loss : -log(e^(prob(i)/sum[(e^prob(i=1 to i=len))]))
    # which is - (log(e^(prob(i))) - log(sum[(e^prob(i=1 to i=len))]))
    # i.e - (prob(i) - log(sum[(e^prob(i=1 to i=len))]))
    
    loss = - (predicted_prob - torch.log(exp_x_sum)) #(batch_size)
    
    return loss.mean() 
    
    