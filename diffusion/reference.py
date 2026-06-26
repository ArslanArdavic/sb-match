import torch

def sample_brownian_state(x0, sigma, t):
    # x0 : [N,3,H,W]
    # t  : [N,]
    sqrt_t = torch.sqrt(t)                                      # [N]
    sqrt_t_broadcast = sqrt_t.view(-1, *(1,) * (x0.dim() - 1))  # [N,1,1,1]
    eps = torch.randn_like(x0)                                  # [N,3,H,W]
    # xt = x0 + sigma * sqrt_t_broadcast * eps : [N,3,H,W]
    return x0 + sigma * sqrt_t_broadcast * eps

def simulate_diffusion(x0,t):
    return path

def sample_intermediate_brownian_bridge(x0,xT,t):
    return xt

def simulate_bridge(x0,xT):
    return path