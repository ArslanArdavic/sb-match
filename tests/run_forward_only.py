import os

import torch
import matplotlib
matplotlib.use("Agg")            # headless node: no display, write PNGs directly
import matplotlib.pyplot as plt

from train import forward_only


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


if __name__ == "__main__":

    #forward_only_train_test()

    forward_only_train_full_size_test()