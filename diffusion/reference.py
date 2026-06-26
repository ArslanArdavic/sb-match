import torch

def sample_brownian_state(x0, sigma, t):
    # x0 : [N,3,H,W]
    # t  : [N,]
    # xt = x0 + sigma * sqrt_t_broadcast * N( 0 , 1) 
    sqrt_t = torch.sqrt(t)                                      # [N]
    sqrt_t_broadcast = sqrt_t.view(-1, *(1,) * (x0.dim() - 1))  # [N,1,1,1]
    eps = torch.randn_like(x0)                                  # [N,3,H,W]
    return x0 + sigma * sqrt_t_broadcast * eps

def sample_conditioned_brownian_bridge(x0, xT, sigma, t):    
    # xt = x0 * (1-t) + xT * t + sigma * N( 0 , t(1-t) )
    # xt = x0 * (1-t) + xT * t + sigma * N( 0 , 1) * sqrt( t(1-t) )

    sqrt_t_times_one_minus_t = torch.sqrt(t * (1-t)).view(-1, *(1,) * (x0.dim() - 1))  # [N,1,1,1]
    
    eps = torch.randn_like(x0) * sqrt_t_times_one_minus_t                              # [N,3,H,W]

    t_broad = t.view(-1, *(1,) * (x0.dim() - 1))                # [N,1,1,1]
    one_minus_t_broad = (1-t).view(-1, *(1,) * (x0.dim() - 1))  # [N,1,1,1]

    return x0 * one_minus_t_broad + xT * t_broad + sigma  * eps

