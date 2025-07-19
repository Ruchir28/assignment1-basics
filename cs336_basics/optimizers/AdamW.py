from math import sqrt
import torch
from typing import Optional, Callable

class AdamW(torch.optim.Optimizer):
    def __init__(self,params,lr=1e-3,betas=(0.9,0.999),eps=1e-8,weight_decay=0.01):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
        }
        super().__init__(params, defaults)
        
    def step(self, closure:Optional[Callable] = None):
        
        loss = None if closure is None else closure()
        
        for grp in self.param_groups:
            lr = grp["lr"]
            beta1, beta2 = grp["betas"]
            eps = grp["eps"]
            weight_decay = grp["weight_decay"]
            
            for p in grp["params"]:
                
                if p.grad is None:
                    continue
                
                state = self.state[p]
                
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p,memory_format=torch.preserve_format)
                    state["v"] = torch.zeros_like(p,memory_format=torch.preserve_format)
                
                m = state["m"]
                v = state["v"]
                t = state["t"] + 1
                
                grad = p.grad.data
                
                # Update biased first moment estimate (momentum)
                # m captures the direction of the update, smoothing out gradients.
                m = beta1 * m + (1 - beta1) * grad

                # Update biased second raw moment estimate (adaptive scaling)
                # v captures the magnitude (uncentered variance) of recent gradients.
                v = beta2 * v + (1 - beta2) * grad**2

                # Bias correction.
                # At the beginning of training, m and v are biased towards zero. (becasue initially we start with zero and 
                # current graidents get very small importance)
                # This correction term adjusts the learning rate to counteract this bias.
                # The formula is a rearrangement of the standard Adam bias correction:
                # m_hat = m / (1 - beta1**t)
                # v_hat = v / (1 - beta2**t)
                # update = lr * m_hat / (sqrt(v_hat) + eps)
                # which can be rewritten as:
                # update = (lr * sqrt(1 - beta2**t) / (1 - beta1**t)) * (m / (sqrt(v) + eps))
                lr_t = lr * (sqrt(1 - (beta2**t)) / (1 - beta1**t))
                
                # The main Adam update rule.
                # The update is the momentum term (m) scaled by the adaptive learning rate (1/sqrt(v)).
                # From momentum term , it smoothen out the direction of updates based on history
                # From variance term it basically tells if the parameter magnitude is small , then it can take a bigger step
                # while if a parameter is large it takes a smaller step , i.e. making sure all of them are being updated acordingly
                # This takes larger steps for parameters with small, consistent gradients and
                # smaller steps for parameters with large or noisy gradients.
                # The `eps` term is for numerical stability.
                p.data = p.data - lr_t * (m / (torch.sqrt(v) + eps))
                
                # Decoupled weight decay (the "W" in AdamW)
                p.data = p.data - lr * weight_decay * p.data
                
                state["t"] = t
                state["m"] = m
                state["v"] = v
