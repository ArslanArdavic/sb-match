import os
import argparse
import time

import torch
import matplotlib
matplotlib.use("Agg")            # headless node: no display, write PNGs directly
import matplotlib.pyplot as plt

from torchvision.utils import save_image
from diffusers import UNet2DModel

from train import forward_only
from utils.process_image import load_afhq_image, load_afhq_val, load_afhq_val


def forward_only_train_test(location="local", resolution=64):

    if location == "local":
        datadir = "/home/arslan/research/literature/foundations-schrodinger-bridges-tang-2026/sb-match/data/afhq/train/"
        outdir = "tests/outputs/forward_only/local/"
    elif location == "lumi":
        datadir = "/flash/project_465002822/sb-match/data/afhq/train/"
        outdir = "/flash/project_465002822/sb-match/tests/outputs/forward_only/lumi/"

    config = {"device"    : "cuda",
              "datadir"   : datadir,
              "downsize"  : resolution ,
              "batch_size": 8,
              "lr"    : 1e-4,
              "epochs": 1,
              "eps"   : 1e-4,
              "sigma" : 1,
            }

    net, step_losses, epoch_losses = forward_only.train(config=config)

    
    os.makedirs(outdir, exist_ok=True)

    # save the net
    torch.save(net.state_dict(), os.path.join(outdir, f"forward_only_{resolution}_net.pt"))

    # plot the step_losses
    plt.figure()
    plt.plot(step_losses)
    plt.xlabel("step"); plt.ylabel("loss"); plt.title(f"forward_only_{resolution} step losses")
    plt.savefig(os.path.join(outdir, f"forward_only_{resolution}_step_losses.png"))
    plt.close()

    # plot the epoch_losses
    plt.figure()
    plt.plot(epoch_losses)
    plt.xlabel("epoch"); plt.ylabel("mean loss"); plt.title(f"forward_only_{resolution} epoch losses")
    plt.savefig(os.path.join(outdir, f"forward_only_{resolution}_epoch_losses.png"))
    plt.close()

def forward_only_sample_trajectory_test(location="local", resolution=64):
    if location == "local":
        image_path ="/home/arslan/research/literature/foundations-schrodinger-bridges-tang-2026/sb-match/data/afhq/val/cat/flickr_cat_000526.jpg"
        model_path = f"/home/arslan/research/literature/foundations-schrodinger-bridges-tang-2026/sb-match/tests/outputs/forward_only/local/forward_only_{resolution}_net.pt"
        outfile = f"tests/outputs/forward_only/local/forward_only_sample_trajectory_test_{resolution}.png"
    elif location == "lumi":
        image_path = "/flash/project_465002822/sb-match/data/afhq/val/cat/flickr_cat_000526.jpg"
        model_path = f"/flash/project_465002822/sb-match/tests/outputs/forward_only/lumi/forward_only_{resolution}_net.pt"
        outfile = f"/flash/project_465002822/sb-match/tests/outputs/forward_only/lumi/forward_only_sample_trajectory_test_{resolution}.png"

    config = {"device"    : "cuda",
              "image_path"   : image_path,
              "downsize"  : resolution,
              "model_path": model_path,
              "N" : 100,
              "sigma" : 1,
            }
    
    device = config["device"]
    N      = config["N"]
    sigma  = config["sigma"]
    dt     = torch.tensor([1 / N], device=device)

    x0 = load_afhq_image(path=config["image_path"], downsize=config["downsize"]).to(device)   # (1,3,64,64), MUST match model res

    if resolution == 64:
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
    elif resolution == 512:
        net = UNet2DModel(
            sample_size=512,
            in_channels=3,
            out_channels=3,
            layers_per_block=4,                          # ref num_res_blocks=4
            block_out_channels=(128, 256, 256, 256),     # nf=128, ch_mult [1,2,2,2]
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
            dropout=0.15,                                # ref dropout 0.15
            norm_num_groups=32,                          # ref GroupNorm
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
    save_image(grid, outfile, nrow=N+1)

# Legacy method
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

# Legacy method
def forward_only_sample_trajectory_full_size_test():
    config = {"device"    : "cuda",
              "datadir"   : "/home/arslan/research/literature/foundations-schrodinger-bridges-tang-2026/sb-match/data/afhq/val/",  
              "downsize"  : 512 ,
              "batch_size": 8 ,
              "model_path": "/home/arslan/research/literature/foundations-schrodinger-bridges-tang-2026/sb-match/tests/outputs/forward_only_512_net.pt",
              "N" : 1000,
              "sigma" : 1,
            }
    
    device = config["device"]
    N      = config["N"]
    sigma  = config["sigma"]
    dt     = torch.tensor([1 / N], device=device)

    x0 = load_afhq_image(downsize=config["downsize"]).to(device)   # (1,3,64,64), MUST match model res

    # build the SAME architecture as the 64px training branch, then load weights
    net = UNet2DModel(
        sample_size=512,
        in_channels=3,
        out_channels=3,
        layers_per_block=4,                          # ref num_res_blocks=4
        block_out_channels=(128, 256, 256, 256),     # nf=128, ch_mult [1,2,2,2]
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        dropout=0.15,                                # ref dropout 0.15
        norm_num_groups=32,                          # ref GroupNorm
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
    save_image(grid, "tests/outputs/forward_only_sample_trajectory_test_512.png", nrow=N+1)


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
    
    parser = argparse.ArgumentParser(description="Run forward-only train/sample tests.")
    parser.add_argument("location",   choices=["local", "lumi"])
    parser.add_argument("resolution", choices=["64", "512"])
    args = parser.parse_args()

    if (args.location, args.resolution) == ("local", "64"):
        forward_only_train_test(location="local", resolution=64)
        forward_only_sample_trajectory_test(location="local", resolution=64)
    elif (args.location, args.resolution) == ("local", "512"):
        forward_only_train_test(location="local", resolution=512)
        forward_only_sample_trajectory_test(location="local", resolution=512)
    elif (args.location, args.resolution) == ("lumi", "64"):
        forward_only_train_test(location="lumi", resolution=64)
        forward_only_sample_trajectory_test(location="lumi", resolution=64)
    elif (args.location, args.resolution) == ("lumi", "512"):
        forward_only_train_test(location="lumi", resolution=512)
        forward_only_sample_trajectory_test(location="lumi", resolution=512)
    else:
        raise ValueError(f"unknown (location, resolution) pair: {(args.location, args.resolution)}")