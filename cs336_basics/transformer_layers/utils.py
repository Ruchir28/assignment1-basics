import os
from typing import IO, BinaryIO, Iterable
import torch
import math
import numpy.typing as npt


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
    
def learning_rate_scheduler(t,lr_max,lr_min,t_warmup,t_cosine_annealing):
    
    if t < t_warmup:
        return lr_max * (t / t_warmup)
    
    
    if t >= t_warmup and t <= t_cosine_annealing:
        cosine_term = (1 + (math.cos(math.pi * (t - t_warmup) / (t_cosine_annealing - t_warmup)))) / 2
        return lr_min + (lr_max - lr_min) * cosine_term
    
    return lr_min
    
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    
    
    gradients = [param.grad for param in parameters if param.grad is not None]
    
    total_norm = torch.norm(torch.stack([torch.norm(g,p=2) for g in gradients]),p=2)
    
    if total_norm > max_l2_norm:
        
        scaling_factor = max_l2_norm / total_norm + 1e-6
        
        for param in parameters:
            if param.grad is not None:
                param.grad.mul_(scaling_factor)
            

def data_loader(dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
    
    total_len = len(dataset)
    
    num_sequences = total_len - context_length
    
    indices = torch.randint(0, num_sequences, (batch_size,))
    
    input_list = []
    output_list = []
    
    for idx in indices:
        i = idx.item()
        input_seq = torch.tensor(dataset[i : i + context_length], dtype=torch.float32)
        output_seq = torch.tensor(dataset[i + 1 : i + context_length + 1], dtype=torch.float32)
        input_list.append(input_seq)
        output_list.append(output_seq)
    
    input_tensor = torch.stack(input_list).to(device)
    output_tensor = torch.stack(output_list).to(device)
    
    return (input_tensor, output_tensor)


def save_checkpoint(model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    obj = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration
    }
    
    torch.save(obj,out)
    

def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer):
    
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    return checkpoint['iteration']


def generate_sequence(
    model: torch.nn.Module,
    sequence: torch.Tensor,  
    max_generated_length: int,
    temperature: float = 1.0,
    top_p: float = 0.9,
):
    # sequence: (B, S) tensor of input token indices
    model.eval()
    generated_seq = sequence
    for _ in range(max_generated_length):
        # Get logits for the last token
        logits = model(generated_seq)  # (B, S, vocab_size)
        last_logits = logits[:, -1, :] / temperature  # (B, vocab_size)

        probs = torch.softmax(last_logits, dim=-1)  # (B, vocab_size)

        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)  # (B, vocab_size)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            # Build mask of tokens to keep
            mask = cumulative_probs <= top_p
            mask[:, 0] = True  # always include top token
            filtered_probs = sorted_probs * mask
            filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
            # Sample in sorted space
            sampled_pos = torch.multinomial(filtered_probs, num_samples=1)  # (B, 1)
            # Map back to original indices
            sampled_token = sorted_indices.gather(dim=-1, index=sampled_pos)  # (B, 1)
        else:
            sampled_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

        generated_seq = torch.cat([generated_seq, sampled_token], dim=1)  # (B, S+1)

    return generated_seq

    
class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self,data: npt.NDArray, context_length: int):
        self.data = data
        self.context_length = context_length
        self.num_sequences = len(data) - context_length

    def __len__(self): 
        return self.num_sequences
    
    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.num_sequences:
            raise IndexError("Index out of bounds")
        
        input_seq = torch.tensor(self.data[idx : idx + self.context_length], dtype=torch.long)
        output_seq = torch.tensor(self.data[idx + 1 : idx + self.context_length + 1], dtype=torch.long)

        return input_seq, output_seq
    
    
