import torch
from torchvision.utils import save_image

from utils.process_image import load_afhq_image
from diffusion.reference import sample_brownian_state, sample_conditioned_brownian_bridge, sample_markovian_drift_target_brownian_bridge


def sample_brownian_state_test(N=5, sigma=1):
    x0 = load_afhq_image()                      # (1, 3, H, W) in [-1, 1]

    states = [x0]
    for i in range(1, N + 1):
        t = torch.tensor([i / N])               # shape (B,) = (1,)
        states.append(sample_brownian_state(x0, sigma, t))

    # (N+1, 3, H, W), undo [-1,1] -> [0,1], lay out as one row
    grid = torch.cat(states, dim=0)
    grid = (grid + 1.0) / 2.0
    save_image(grid, f"tests/outputs/brownian_states_sigma_{sigma}.png", nrow=N + 1)


def sample_brownian_state_parallel_test(N=5, sigma=1):
    x0 = load_afhq_image()                      # (1, 3, H, W) in [-1, 1]

    x_stack = x0.repeat(N, 1, 1, 1)                          # (N, 3, H, W)
    t = torch.tensor([i / N for i in range(1, N + 1)])       # (N,)
    states = sample_brownian_state(x_stack, sigma, t)        # (N, 3, H, W)

    grid = torch.cat([x0, states], dim=0)                    # (N+1, 3, H, W)
    grid = (grid + 1.0) / 2.0                                # [-1,1] -> [0,1]
    save_image(grid, f"tests/outputs/brownian_states_sigma_{sigma}_parallel.png", nrow=N + 1)


def sample_conditioned_brownian_bridge_test(N=5, sigma=1):
    x0 = load_afhq_image()                      # (1, 3, H, W) in [-1, 1]
    xT = load_afhq_image(path="/home/arslan/research/literature/foundations-schrodinger-bridges-tang-2026/sb-match/data/afhq/val/wild/pixabay_wild_000840.jpg")                      

    states = [x0]
    for i in range(1, N):                       # t  in [0.2, 0.4, 0.6, 0.8] by default
        t = torch.tensor([i / N])           
        states.append(sample_conditioned_brownian_bridge(x0, xT, sigma, t))    
    states.append(xT)

    grid = torch.cat(states, dim=0)
    grid = (grid + 1.0) / 2.0
    save_image(grid, f"tests/outputs/conditioned_brownian_bridge_sigma_{sigma}.png", nrow=N + 1)


def sample_conditioned_brownian_bridge_parallel_test(N=5, sigma=1):
    x0 = load_afhq_image(downsize=512)                      # (1, 3, H, W) in [-1, 1]
    xT = load_afhq_image(
        path="/home/arslan/research/literature/foundations-schrodinger-bridges-tang-2026/sb-match/data/afhq/val/wild/pixabay_wild_000840.jpg",
        downsize=512
        )

    x0_stack = x0.repeat(N-1, 1, 1, 1)                          # (N-1, 3, H, W)
    xT_stack = xT.repeat(N-1, 1, 1, 1)                          # (N-1, 3, H, W)
    t = torch.tensor([i / N for i in range(1, N)])              # (N-1,)
    states = sample_conditioned_brownian_bridge(x0_stack, xT_stack, sigma, t)

    grid = torch.cat([x0, states, xT], dim=0)                    # (N+1, 3, H, W)
    grid = (grid + 1.0) / 2.0                                # [-1,1] -> [0,1]
    save_image(grid, f"tests/outputs/conditioned_brownian_bridge_sigma_{sigma}_parallel.png", nrow=N + 1)


def sample_markovian_drift_target_brownian_bridge_test(N, sigma=1):
    # Simulate SDE starting from x0 with instantenous vt drift  
    
    x0 = load_afhq_image(downsize=512)                      # (1, 3, H, W) in [-1, 1]
    xT = load_afhq_image(
        path="/home/arslan/research/literature/foundations-schrodinger-bridges-tang-2026/sb-match/data/afhq/val/wild/pixabay_wild_000840.jpg",
        downsize=512
        )
    
    states = [x0]
    xt = x0
    dt = torch.tensor([1 / N])
    for i in range(N): 
        t = torch.tensor([i / N])       # t in [0, 1/N, ... , (N-1)/N]
        vt = sample_markovian_drift_target_brownian_bridge(xt, xT, t)
        xt = xt + vt * dt + sigma * torch.sqrt(dt) * torch.randn_like(xt) 
        states.append(xt)

    grid = torch.cat(states, dim=0)
    grid = (grid + 1.0) / 2.0
    save_image(grid, f"tests/outputs/markovian_drift_target_brownian_bridge_sigma_{sigma}.png", nrow=N + 1)


if __name__ == "__main__":
    #for sigma in [0.1, 0.5, 1, 2]:
    #    sample_brownian_state_test(sigma=sigma)
    #sample_brownian_state_parallel_test(sigma=0.5)

    #for sigma in [0.1, 0.5, 1, 2]:
    #    sample_conditioned_brownian_bridge_test(sigma=sigma)
    #sample_conditioned_brownian_bridge_parallel_test(sigma=1)

    sample_markovian_drift_target_brownian_bridge_test(N=20, sigma=1)
    