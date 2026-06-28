import time

import torch
from torch.utils.data import TensorDataset, DataLoader

from diffusers import UNet2DModel

from utils.process_image import load_afhq_train
from diffusion.reference import sample_conditioned_brownian_bridge, sample_markovian_drift_target_brownian_bridge


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def train(config):

    PRM = ["device", "datadir", "downsize", "batch_size", "sample_batch_size", "gradient_cp", "lr", "N", "n_outer", "epochs_per_drift", "eps", "sigma"]
    assert not [prm for prm in PRM if prm not in config]

    config["downsize"] = 64  # This is a prototype script working on small scale.
    device = config["device"]
    N      = config["N"]
    sigma  = config["sigma"]

    dt = torch.tensor([1 / N], device=device) # EM simulation needs

    # Only used while sampling - maximize the batch size to 32
    x0_dataloader = load_afhq_train(datadir=config["datadir"], cls="cat", downsize=config["downsize"], batch_size=config["sample_batch_size"], drop_last=False)
    xT_dataloader = load_afhq_train(datadir=config["datadir"], cls="wild", downsize=config["downsize"], batch_size=config["sample_batch_size"], drop_last=False)
    
    X0 = torch.cat([x for (x,_) in x0_dataloader])      
    XT = torch.cat([x for (x,_) in xT_dataloader])      
    m = min(len(X0), len(XT))
    coupling = TensorDataset(X0[:m], XT[:m]) # Make sizes match, shuffled already

    forward_net = UNet2DModel(
        sample_size=config["downsize"],            # 64
        in_channels=3,
        out_channels=3,
        layers_per_block=2,                        # ref uses 4; 2 is plenty to validate
        block_out_channels=(128, 256, 256, 256),   # nf=128, ch_mult [1,2,2,2]
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    ).to(device)
    
    backward_net = UNet2DModel(
        sample_size=config["downsize"],            # 64
        in_channels=3,
        out_channels=3,
        layers_per_block=2,                        # ref uses 4; 2 is plenty to validate
        block_out_channels=(128, 256, 256, 256),   # nf=128, ch_mult [1,2,2,2]
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    ).to(device)
    
    if config["gradient_cp"]: 
        forward_net.enable_gradient_checkpointing()
    forward_net.train()

    if config["gradient_cp"]: 
        backward_net.enable_gradient_checkpointing()
    backward_net.train()

    criterion = torch.nn.MSELoss() 

    forward_optimizer  = torch.optim.Adam(params=forward_net.parameters(), lr=config["lr"])
    backward_optimizer = torch.optim.Adam(params=backward_net.parameters(), lr=config["lr"])

    losses = { 
        "forward"  : {
            "epoch" : [],
            "step"  : [],
            },
        "backward" : {
            "epoch" : [],
            "step"  : [],
            },
        }

    for outer in range(config["n_outer"]):

        for direction in ("forward" , "backward"):
            
            if direction == "forward":

                # Train
                for epoch in range(config["epochs_per_drift"]):

                    running, n = 0.0, 0  # compute average epoch loss
                    
                    log(f"[{direction} {outer}] epoch {epoch} - training")

                    for x0, xT in DataLoader(coupling, batch_size=config["batch_size"], shuffle=True):
                        x0, xT = x0.to(device), xT.to(device)
                        t = torch.rand(x0.shape[0], device=device) * (1 - 2*config["eps"]) + config["eps"]
                        xt = sample_conditioned_brownian_bridge(x0, xT, config["sigma"], t)
                        target = sample_markovian_drift_target_brownian_bridge(xt, xT, t)

                        pred = forward_net(xt, t * 1000).sample 
                        loss = criterion(pred, target)
                        forward_optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(forward_net.parameters(), 1.0)
                        forward_optimizer.step()

                        l = loss.item()
                        losses["forward"]["step"].append(l)
                        running += l; n += 1

                    epoch_loss = running / max(n, 1)
                    losses["forward"]["epoch"].append(epoch_loss)
                    log(f"[{direction} {outer}] epoch {epoch} - average_loss {epoch_loss:.4f}")
                    
                
                # Sample 
                log(f"[{direction} {outer}] - sampling ")
                    
                X0 = []
                XT_sim = []
                with torch.no_grad():
                    for x0, _ in x0_dataloader:
                        xt = x0.to(device)
                        for i in range(N):
                            t = torch.full((xt.shape[0],), i / N, device=device)
                            vt = forward_net(xt, t * 1000).sample 
                            xt = xt + vt * dt + sigma * torch.sqrt(dt) * torch.randn_like(xt)

                        X0.append(x0.cpu())   
                        XT_sim.append(xt.cpu())  

                coupling = TensorDataset(torch.cat(X0), torch.cat(XT_sim))
                
            elif direction == "backward":

                # Train
                for epoch in range(config["epochs_per_drift"]):

                    running, n = 0.0, 0  # compute average epoch loss

                    log(f"[{direction} {outer}] epoch {epoch} - training")

                    for x0, xT in DataLoader(coupling, batch_size=config["batch_size"], shuffle=True):
                        x0, xT = x0.to(device), xT.to(device)
                        t = torch.rand(x0.shape[0], device=device) * (1 - 2*config["eps"]) + config["eps"]
                        xt = sample_conditioned_brownian_bridge(x0=xT, xT=x0, sigma=config["sigma"], t=t) 
                        target = sample_markovian_drift_target_brownian_bridge(xt, xT=x0, t=t)
                        
                        pred = backward_net(xt, t * 1000).sample 
                        
                        loss = criterion(pred, target)
                        backward_optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(backward_net.parameters(), 1.0)
                        backward_optimizer.step()

                        l = loss.item()
                        losses["backward"]["step"].append(l)
                        running += l; n += 1

                    epoch_loss = running / max(n, 1)
                    losses["backward"]["epoch"].append(epoch_loss)
                    log(f"[{direction} {outer}] epoch {epoch} - average_loss {epoch_loss:.4f}")

                # Sample 
                log(f"[{direction} {outer}] - sampling ")

                X0_sim = []
                XT     = []
                with torch.no_grad():
                    for xT, _ in xT_dataloader:
                        xt = xT.to(device)
                        for i in range(N):
                            t = torch.full((xt.shape[0],), i / N, device=device)
                            vt = backward_net(xt, t * 1000).sample 
                            xt = xt + vt * dt + sigma * torch.sqrt(dt) * torch.randn_like(xt)
                        
                        X0_sim.append(xt.cpu())
                        XT.append(xT.cpu())

                coupling = TensorDataset(torch.cat(X0_sim), torch.cat(XT))

    return forward_net, backward_net, losses
        