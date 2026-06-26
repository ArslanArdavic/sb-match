import torch
from torchvision.utils import save_image

from utils.process_image import load_afhq_image
from diffusion.reference import sample_brownian_state


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


if __name__ == "__main__":
    for sigma in [0.1, 0.5, 1, 2]:
        sample_brownian_state_test(sigma=sigma)
    sample_brownian_state_parallel_test(sigma=0.5)