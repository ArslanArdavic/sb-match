import os 

import torch
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

from train import forward_backward_prototype

def forward_backward_prototype_test():

    config = {
          "device"    : "cuda",
          "datadir"   : "/home/arslan/research/literature/foundations-schrodinger-bridges-tang-2026/sb-match/data/afhq/train/",  
          "downsize"  : 64 ,
          "batch_size": 8 ,
          "sample_batch_size": 32 ,
          "lr"    : 1e-4,
          "N" : 8,
          "n_outer" : 2, 
          "epochs_per_drift": 2,
          "eps"   : 1e-4,
          "sigma" : 1,
        }

    
    forward_net, backward_net, losses = forward_backward_prototype.train(config)

    outdir = "tests/outputs"
    os.makedirs(outdir, exist_ok=True)

    # save nets
    torch.save(forward_net.state_dict(), os.path.join(outdir, "prototype_forward_64_net.pt"))
    torch.save(backward_net.state_dict(), os.path.join(outdir, "prototype_backward_64_net.pt"))

    # plot the step_losses
    plt.figure()
    plt.plot(losses["forward"]["step"])
    plt.xlabel("step"); plt.ylabel("loss"); plt.title("prototype_forward_64 step losses")
    plt.savefig(os.path.join(outdir, "prototype_forward_64_step_losses.png"))
    plt.close()

    plt.figure()
    plt.plot(losses["backward"]["step"])
    plt.xlabel("step"); plt.ylabel("loss"); plt.title("prototype_backward_64 step losses")
    plt.savefig(os.path.join(outdir, "prototype_backward_64_step_losses.png"))
    plt.close()

    # plot the epoch_losses
    plt.figure()
    plt.plot(losses["forward"]["epoch"])
    plt.xlabel("epoch"); plt.ylabel("mean loss"); plt.title("prototype_forward_64 epoch losses")
    plt.savefig(os.path.join(outdir, "prototype_forward_64_epoch_losses.png"))
    plt.close()

    plt.figure()
    plt.plot(losses["backward"]["epoch"])
    plt.xlabel("epoch"); plt.ylabel("mean loss"); plt.title("prototype_backward_64 epoch losses")
    plt.savefig(os.path.join(outdir, "prototype_backward_64_epoch_losses.png"))
    plt.close()

def forward_backward_prototype_hpc_test():

    config = {
          "device"    : "cuda",
          "datadir"   : "/flash/project_465002822/sb-match/data/afhq/train/",  
          "downsize"  : 64 ,
          "batch_size": 64 ,
          "sample_batch_size": 512 ,
          "lr"    : 1e-4,
          "N" : 8,
          "n_outer" : 2, 
          "epochs_per_drift": 2,
          "eps"   : 1e-4,
          "sigma" : 1,
        }

    forward_net, backward_net, losses = forward_backward_prototype.train(config)

    outdir = "/flash/project_465002822/sb-match/tests/outputs"
    os.makedirs(outdir, exist_ok=True)

    # save nets
    torch.save(forward_net.state_dict(), os.path.join(outdir, "prototype_forward_64_hpc_net.pt"))
    torch.save(backward_net.state_dict(), os.path.join(outdir, "prototype_backward_64_hpc_net.pt"))

    # plot the step_losses
    plt.figure()
    plt.plot(losses["forward"]["step"])
    plt.xlabel("step"); plt.ylabel("loss"); plt.title("prototype_forward_64 step losses")
    plt.savefig(os.path.join(outdir, "prototype_forward_64_hpc_step_losses.png"))
    plt.close()

    plt.figure()
    plt.plot(losses["backward"]["step"])
    plt.xlabel("step"); plt.ylabel("loss"); plt.title("prototype_backward_64 step losses")
    plt.savefig(os.path.join(outdir, "prototype_backward_64_hpc_step_losses.png"))
    plt.close()

    # plot the epoch_losses
    plt.figure()
    plt.plot(losses["forward"]["epoch"])
    plt.xlabel("epoch"); plt.ylabel("mean loss"); plt.title("prototype_forward_64 epoch losses")
    plt.savefig(os.path.join(outdir, "prototype_forward_64_hpc_epoch_losses.png"))
    plt.close()

    plt.figure()
    plt.plot(losses["backward"]["epoch"])
    plt.xlabel("epoch"); plt.ylabel("mean loss"); plt.title("prototype_backward_64 epoch losses")
    plt.savefig(os.path.join(outdir, "prototype_backward_64_hpc_epoch_losses.png"))
    plt.close()


if __name__ == "__main__":

    forward_backward_prototype_test()