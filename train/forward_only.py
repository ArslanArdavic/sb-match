import torch
from diffusers import UNet2DModel

from utils.process_image import load_afhq_train, load_afhq_val
from diffusion.reference import sample_conditioned_brownian_bridge, sample_markovian_drift_target_brownian_bridge


def train(config):
    
    PRM = ["device", "datadir", "downsize", "batch_size", "lr", "epochs", "eps", "sigma"]
    assert not [prm for prm in PRM if prm not in config]

    device = config["device"]

    x0_dataloader = load_afhq_train(datadir=config["datadir"], cls="cat", downsize=config["downsize"], batch_size=config["batch_size"])
    xT_dataloader = load_afhq_train(datadir=config["datadir"], cls="wild", downsize=config["downsize"], batch_size=config["batch_size"])

    if config["downsize"] == 64:
        net = UNet2DModel(
            sample_size=config["downsize"],            # 64
            in_channels=3,
            out_channels=3,
            layers_per_block=2,                        # ref uses 4; 2 is plenty to validate
            block_out_channels=(128, 256, 256, 256),   # nf=128, ch_mult [1,2,2,2]
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        ).to(device)
    elif config["downsize"] == 512:
        # Faithful to the reference DDPMpp config (conf/model/DDPMpp_32.yaml +
        # conf/dataset/afhq_transfer.yaml): nf=128, ch_mult [1,2,2,2], num_res_blocks=4,
        # dropout 0.15, GroupNorm. NOTE: attn_resolutions=[16] is NOT reached at 512 with
        # 4 levels (512->256->128->64), so — exactly as the reference behaves — there are
        # NO attention blocks here.
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

    net.enable_gradient_checkpointing()
    net.train()

    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(params=net.parameters(), lr=config["lr"])

    step_losses = []     # loss after every step
    epoch_losses = []    # mean loss per epoch
    for epoch in range(config["epochs"]):
        running, n = 0.0, 0
        for (x0, _), (xT, _) in zip(x0_dataloader, xT_dataloader):
            x0, xT = x0.to(device), xT.to(device)
            t = torch.rand(x0.shape[0], device=device) * (1 - 2*config["eps"]) + config["eps"]
            xt = sample_conditioned_brownian_bridge(x0, xT, config["sigma"], t)
            target = sample_markovian_drift_target_brownian_bridge(xt, xT, t)

            pred = net(xt, t * 1000).sample          # scale t [0,1] -> [0,1000] for the time embedding

            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)   # ref: grad_clip=1.0; tames t->1 target tail
            optimizer.step()

            l = loss.item()
            step_losses.append(l)
            running += l; n += 1

        epoch_losses.append(running / max(n, 1))
        print(f"epoch {epoch}  mean_loss {epoch_losses[-1]:.4f}")

    return net, step_losses, epoch_losses
    

def sample(config):

    PRM = ["device", "datadir", "downsize", "batch_size", "model_path", "N", "sigma"]
    assert not [prm for prm in PRM if prm not in config]

    device = config["device"]

    x0_dataloader = load_afhq_val(datadir=config["datadir"], cls="cat", downsize=config["downsize"], batch_size=config["batch_size"])
    
    # load net from the config["model_path"]
    N     = config["N"]
    sigma = config["sigma"]
    dt    = torch.tensor([1 / N])

    xT = []

    for (x0, _) in x0_dataloader:
        xt = x0
        for i in range(N):
            t = torch.tensor([i / N]) 
            vt = net(xt, t*1000).sample
            xt = xt + vt * dt + sigma * torch.sqrt(dt) * torch.randn_like(xt) 
        xT = torch.cat([xT , xt] , dim=0)

    return xT