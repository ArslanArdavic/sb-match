import os
import time

import torch
import matplotlib
matplotlib.use("Agg")            # headless node: no display, write PNGs directly
import matplotlib.pyplot as plt

from torchvision.utils import save_image
from diffusers import UNet2DModel

from train import forward_only
from utils.process_image import load_afhq_image, load_afhq_val, load_afhq_val


def forward_only_train_test():

    config = {"device"    : "cuda",
              "datadir"   : "/home/arslan/research/literature/foundations-schrodinger-bridges-tang-2026/sb-match/data/afhq/train/",  
              "downsize"  : 64 ,
              "batch_size": 8 ,
              "lr"    : 1e-4,
              "epochs": 10,
              "eps"   : 1e-4,
              "sigma" : 1,
            }

    net, step_losses, epoch_losses = forward_only.train(config=config)

    outdir = "tests/outputs"
    os.makedirs(outdir, exist_ok=True)

    # save the net
    torch.save(net.state_dict(), os.path.join(outdir, "forward_only_64_net.pt"))

    # plot the step_losses
    plt.figure()
    plt.plot(step_losses)
    plt.xlabel("step"); plt.ylabel("loss"); plt.title("forward_only_64 step losses")
    plt.savefig(os.path.join(outdir, "forward_only_64_step_losses.png"))
    plt.close()

    # plot the epoch_losses
    plt.figure()
    plt.plot(epoch_losses)
    plt.xlabel("epoch"); plt.ylabel("mean loss"); plt.title("forward_only_64 epoch losses")
    plt.savefig(os.path.join(outdir, "forward_only_64_epoch_losses.png"))
    plt.close()


def forward_only_train_full_size_test():

    config = {"device"    : "cuda",
              "datadir"   : "/flash/project_465002822/sb-match/data/afhq/train/",  
              "downsize"  : 512 ,
              "batch_size": 4 ,
              "lr"    : 1e-4,
              "epochs": 20,
              "eps"   : 1e-4,
              "sigma" : 1,
            }

    net, step_losses, epoch_losses = forward_only.train(config=config)

    outdir = "/flash/project_465002822/sb-match/tests/outputs"
    os.makedirs(outdir, exist_ok=True)

    # save the net
    torch.save(net.state_dict(), os.path.join(outdir, "forward_only_512_net.pt"))

    # plot the step_losses
    plt.figure()
    plt.plot(step_losses)
    plt.xlabel("step"); plt.ylabel("loss"); plt.title("forward_only_512 step losses")
    plt.savefig(os.path.join(outdir, "forward_only_512_step_losses.png"))
    plt.close()

    # plot the epoch_losses
    plt.figure()
    plt.plot(epoch_losses)
    plt.xlabel("epoch"); plt.ylabel("mean loss"); plt.title("forward_only_512 epoch losses")
    plt.savefig(os.path.join(outdir, "forward_only_512_epoch_losses.png"))
    plt.close()


def forward_only_sample_trajectory_test():
    config = {"device"    : "cuda",
              "datadir"   : "/home/arslan/research/literature/foundations-schrodinger-bridges-tang-2026/sb-match/data/afhq/val/",  
              "downsize"  : 64 ,
              "batch_size": 8 ,
              "model_path": "/home/arslan/research/literature/foundations-schrodinger-bridges-tang-2026/sb-match/tests/outputs/forward_only_64_net.pt",
              "N" : 2000,
              "sigma" : 1,
            }
    
    device = config["device"]
    N      = config["N"]
    sigma  = config["sigma"]
    dt     = torch.tensor([1 / N], device=device)

    x0 = load_afhq_image(downsize=config["downsize"]).to(device)   # (1,3,64,64), MUST match model res

    # build the SAME architecture as the 64px training branch, then load weights
    net = UNet2DModel(
        sample_size=config["downsize"],
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 256, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    ).to(device)
    net.load_state_dict(torch.load(config["model_path"], map_location=device))
    net.eval()

    states = [x0]
    xt = x0
    with torch.no_grad():
        for i in range(N):
            t = torch.tensor([i / N], device=device)              # current time, 0 .. (N-1)/N
            vt = net(xt, t * 1000).sample                          # MARGINAL drift, no xT; same t*1000 as training
            xt = xt + vt * dt + sigma * torch.sqrt(dt) * torch.randn_like(xt)
            states.append(xt)

    grid = (torch.cat(states, dim=0).cpu() + 1.0) / 2.0            # all N+1 frames
    save_image(grid, "tests/outputs/forward_only_sample_trajectory_test_64.png", nrow=N+1)


def forward_only_sample_val_terminal_test():
    config = {"device"    : "cuda",
              "datadir"   : "/home/arslan/research/literature/foundations-schrodinger-bridges-tang-2026/sb-match/data/afhq/train/",  
              "downsize"  : 64 ,
              "batch_size": 32 ,
              "model_path": "/home/arslan/research/literature/foundations-schrodinger-bridges-tang-2026/sb-match/tests/outputs/forward_only_64_net.pt",
              "N" : 30,
              "sigma" : 1,
            }
    
    device = config["device"]
    N      = config["N"]
    sigma  = config["sigma"]
    dt     = torch.tensor([1 / N], device=device)

    x0_dataloader = load_afhq_val(datadir=config["datadir"], cls="cat", downsize=config["downsize"], batch_size=config["batch_size"])
    
    # build the SAME architecture as the 64px training branch, then load weights
    net = UNet2DModel(
        sample_size=config["downsize"],
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 256, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    ).to(device)
    net.load_state_dict(torch.load(config["model_path"], map_location=device))
    net.eval()

    total = len(x0_dataloader.dataset)                               # images to simulate forward
    done  = 0
    t_start = time.perf_counter()

    x0_stack = []
    xT_stack = []
    with torch.no_grad():
        for (x0, _) in x0_dataloader:
            x0 = x0.to(device)
            xt = x0
            for i in range(N):
                t = torch.full((xt.shape[0],), i / N, device=device)  # (B,) current time, 0 .. (N-1)/N
                vt = net(xt, t * 1000).sample                          # MARGINAL drift, no xT; same t*1000 as training
                xt = xt + vt * dt + sigma * torch.sqrt(dt) * torch.randn_like(xt)
            x0_stack.append(x0.cpu())
            xT_stack.append(xt.cpu())

            done += x0.shape[0]
            print(f"simulated {done}/{total} images  ({time.perf_counter() - t_start:.1f}s)")

    print(f"done: {total} images in {time.perf_counter() - t_start:.1f}s")

    X0 = torch.cat(x0_stack, dim=0)                               # (M, 3, H, W) input cats
    XT = torch.cat(xT_stack, dim=0)                               # (M, 3, H, W) sampled wilds
    pairs = torch.stack([X0, XT], dim=1).reshape(-1, *X0.shape[1:])  # x0_0, xT_0, x0_1, xT_1, ...
    grid = (pairs + 1.0) / 2.0
    save_image(grid, "tests/outputs/forward_only_sample_val_terminal_test_64.png", nrow=2)  # 2 cols: cat | wild

if __name__ == "__main__":

    #forward_only_train_test()

    #forward_only_train_full_size_test()

    #forward_only_sample_trajectory_test()

    forward_only_sample_val_terminal_test()