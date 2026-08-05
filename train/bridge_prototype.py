
"""pseudocode for training a bridge model with the caching mechanism

################################################################################################################################

OUR DESING CHOICES:
    1. ALIGNS
    2. ALIGNS
    3. ALIGNS
    4. EFFECTIVELY ALINGS: drops the old handle and empties the cuda cache after the outer != 0 check
    5. EFFECTIVELY ALINGS: indepented coupling for the first outer iteration on both directions
    6. EFFECTIVELY ALINGS: one N for cache simulation and for evaluation
    7. EFFECTIVELY ALINGS: one training tuple per cached pair per visit
    8. cache lives in cpu memory, gone on restart
    9. no resume, a killed job re-simulates the whole cache
    10. simulate the cache with the live training net
    11. cache_npair and steps_cache_refresh set directly, nothing logged


REFERENCE CHOICES:
    1. first train backward
    2. first iteration does not use cache
    3. cache_npar // cache_batch_size separate draws at cache_batch_size (exact division asserted),
       so cache size is not capped by dataset size
    4. drops the old handle and empties the cuda cache before simulating a new one
    5. parametrized to allow brownian motion couple, ind in first outer for both directions in afqh setup
    6. cache_num_steps separate from num_steps / test_num_steps (left equal at afhq)
    7. num_repeat_data repeat_interleaves each cached pair inside the batch, more noise draws at no extra nfe (1 at afhq)
    8. cache written to disk as an .npy memmap per (direction, outer), read lazily, old ones pruned
    9. per-batch temp .pt + seed derived from (outer, refresh_idx, batch) + completion marker make a
       refresh resumable and bit-identical, next stage pre-cached right after the checkpoint
    10. simulate with an ema copy of the net in eval mode
    11. cache_npar derived from batch_size * stride / num_repeat_data by default, cache_epochs and data_epochs logged at startup

################################################################################################################################
"""
import wandb
import torch
from torch.utils.data import TensorDataset, DataLoader
from diffusers import UNet2DModel

from utils.process_image import load_afhq_train, infinite_dataloader
from diffusion.reference import sample_conditioned_brownian_bridge, sample_markovian_drift_target_brownian_bridge

def train(config):

    PRM = ["device", "datadir", "downsize", "cache_limit", "batch_size", "lr", "n_outer", "steps_per_drift",
           "steps_cache_refresh", "cache_npair", "N", "sigma", "eps", "log_stride"]
    assert not [prm for prm in PRM if prm not in config]

    device              = config["device"]
    datadir             = config["datadir"]
    downsize            = config["downsize"] 
    cache_limit         = config["cache_limit"] 
    batch_size          = config["batch_size"]
    lr                  = config["lr"]
    n_outer             = config["n_outer"]
    steps_per_drift     = config["steps_per_drift"]
    steps_cache_refresh = config["steps_cache_refresh"]
    cache_npair         = config["cache_npair"]
    N                   = config["N"]
    sigma               = config["sigma"]
    eps                 = config["eps"]
    log_stride          = config["log_stride"]

    assert cache_npair % cache_limit == 0
    assert cache_npair % batch_size  == 0


    run = setup_wandb(config)
    run.config.update({
        "cache_epochs": batch_size * steps_cache_refresh / cache_npair,
        "data_epochs":  steps_per_drift * cache_npair / (5153 * steps_cache_refresh),
    })

    dt = torch.tensor([1 / N], device=device) # EM simulation needs
    
    infinite_source_cache_dataloader = infinite_dataloader(load_afhq_train(datadir=datadir, cls="cat",  downsize=downsize, batch_size=cache_limit, drop_last=True))
    infinite_target_cache_dataloader = infinite_dataloader(load_afhq_train(datadir=datadir, cls="wild", downsize=downsize, batch_size=cache_limit, drop_last=True))

    infinite_source_train_dataloader = infinite_dataloader(load_afhq_train(datadir=datadir, cls="cat",  downsize=downsize, batch_size=batch_size, drop_last=True))
    infinite_target_train_dataloader = infinite_dataloader(load_afhq_train(datadir=datadir, cls="wild", downsize=downsize, batch_size=batch_size, drop_last=True))

    forward_net, backward_net = build_models(downsize=downsize)
    forward_net.to(device)
    backward_net.to(device)

    forward_optimizer  = torch.optim.Adam(params=forward_net.parameters(),lr=lr)
    backward_optimizer = torch.optim.Adam(params=backward_net.parameters(),lr=lr)

    criterion = torch.nn.MSELoss()

    for outer in range(n_outer):

        for direction in ("bacward", "forward"):

            if direction == "forward":          

                forward_net.train()

                step = 0

                while step < steps_per_drift:

                    if outer != 0 and step % steps_cache_refresh == 0:

                        infinite_pair_loader, X0_sim, XT = None, None, None
                        torch.cuda.empty_cache()

                        X0_sim = torch.empty(cache_npair, 3, downsize, downsize) 
                        XT     = torch.empty(cache_npair, 3, downsize, downsize) 

                        backward_net.eval()

                        for i in range(cache_npair // cache_limit):
                            xT, _ = next(infinite_target_cache_dataloader) # dismiss the label
                            
                            # Sample
                            with torch.no_grad():
                                xt = xT.to(device)
                                for j in range(N):
                                    t = torch.full((xt.shape[0],), j / N, device=device)
                                    vt = backward_net(xt, t * 1000).sample 
                                    xt = xt + vt * dt + sigma * torch.sqrt(dt) * torch.randn_like(xt)

                            X0_sim[i*cache_limit : (i+1)*cache_limit] = xt.cpu()
                            XT[i*cache_limit : (i+1)*cache_limit]     = xT

                        infinite_pair_loader =  infinite_dataloader(DataLoader(TensorDataset(X0_sim, XT), batch_size=batch_size, shuffle=True, drop_last=True))

                    if outer == 0:
                        x0, _ = next(infinite_source_train_dataloader)
                        xT, _ = next(infinite_target_train_dataloader)
                    else:
                        x0, xT = next(infinite_pair_loader)

                    # Update forward_net 
                    x0, xT  = x0.to(device), xT.to(device)
                    t = torch.rand(x0.shape[0], device=device) * (1 - 2*eps) + eps
                    xt = sample_conditioned_brownian_bridge(x0, xT, sigma, t)
                    target = sample_markovian_drift_target_brownian_bridge(xt, xT, t)

                    pred = forward_net(xt, t * 1000).sample 
                    loss = criterion(pred, target)
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(forward_net.parameters(), 1.0) # pre-clip gradient norm
                    forward_optimizer.step()
                    forward_optimizer.zero_grad()

                    if step % log_stride == 0 :
                        # no step= kwarg: wandb's commit counter advances on its own, global_step is the x axis
                        run.log({
                            f"{direction}/loss"      : loss.item(),
                            f"{direction}/grad_norm" : grad_norm.item(),
                            "global_step"            : outer * steps_per_drift + step,
                        })

                    step += 1

    run.finish()
            
def build_models(downsize):

    if downsize == 64:
        # ref: DDPMpp_32.yaml (nf=128, ch_mult [1,2,2,2], num_res_blocks=4, attn_resolutions=[16], dropout=0.15)
        # 4 levels -> resolutions 64, 32, 16, 8 ; attn at 16 (down idx 2 / up idx 1)
        kwargs = dict(
            sample_size=downsize, # 64
            in_channels=3,
            out_channels=3,
            layers_per_block=4,                        # ref uses 4; 2 is plenty to validate
            block_out_channels=(128, 256, 256, 256),   # nf=128, ch_mult [1,2,2,2]
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
            dropout=0.15,
        )

    elif downsize == 512:
        # ref switches config at 512: DDPMpp_RF.yaml (nf=128, ch_mult [1,1,2,2,2,2,2], num_res_blocks=2, attn_resolutions=[16], dropout=0)
        # 7 levels -> resolutions 512, 256, 128, 64, 32, 16, 8 ; attn at 16 (down idx 5 / up idx 1)
        kwargs = dict(
            sample_size=downsize, # 512
            in_channels=3,
            out_channels=3,
            layers_per_block=2,                                       # ref drops 4 -> 2 to pay for the three extra levels
            block_out_channels=(128, 128, 256, 256, 256, 256, 256),   # nf=128, ch_mult [1,1,2,2,2,2,2]
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
            dropout=0.0,
        )

    else:
        raise ValueError(f"Parameter not allowed. downsize : {downsize}")

    forward_net  = UNet2DModel(**kwargs)
    backward_net = UNet2DModel(**kwargs)

    return forward_net, backward_net

def setup_wandb(config):

    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="arda-arslan-allab",
        # Set the wandb project where this run will be logged.
        project="dsbm-afqh",
        # Track hyperparameters and run metadata.
        config=config,
    )

    # Plot everything against global_step instead of wandb's own commit counter. Both directions of an
    # outer share a global_step range, so each series comes out continuous rather than a 50% duty cycle comb.
    run.define_metric("global_step")
    run.define_metric("*", step_metric="global_step")

    return run


"""

infinite_source_dataloader = build_infinite_dataloader(batch_size=cache_limit, data=source)
infinite_target_dataloader = build_infinite_dataloader(batch_size=cache_limit, data=target)

infinite_source_train_dataloader = build_infinite_dataloader(batch_size=batch_size, data=source)
infinite_target_train_dataloader = build_infinite_dataloader(batch_size=batch_size, data=target)

forward_net , backward_net = build_models
define loss function and build_optimizers


##################### REFERENCE STRUCTURE with OUR DESIGN CHOICES #####################                 

for outer in range(n_outer):

    for direction in ("bacward", "forward"):

        if direction == "forward":        
        
            step = 0

            while step < steps_per_drift:
            
                if step % steps_cache_refresh == 0:
                                    

                    if outer != 0:
                        
                        release infinite_pair_loader, x0 and xT
                        empty cuda cache

                        for i in range(cache_npair // cache_limit)
                        
                            xT = infinite_target_dataloader.get_batch_and_proceed
                            x0 = start from xT sample with the backward_net with no gradient
                            store x0 and xT in the cpu
                        
                        couple accumulated x0 and xT                    
                        
                        infinite_pair_loader = build_infinite_dataloader(batch_size=batch_size, data=coupling)

                if outer == 0 :
                    
                    x0 = infinite_source_train_dataloader.get_batch_and_proceed
                    xT = infinite_target_train_dataloader.get_batch_and_proceed

                    batch = coupled x0 and xT
                else:
                    batch = next(infinite_pair_loader)
                    
                update forward_net using the batch
                step += 1

##################### CUSTOM STRUCTURE with OUR DESIGN CHOICES #####################   
                
for outer in range(n_outer):

    for direction in ("bacward", "forward"):

        if direction == "forward":        
        
            step = 0

            while step < steps_per_drift:
            
                if outer == 0:
                
                    release infinite_pair_loader, x0 and xT

                    x0 = infinite_source_train_dataloader.get_batch_and_proceed
                    xT = infinite_target_train_dataloader.get_batch_and_proceed

                    couple x0 and xT
                    update forward_net using the coupling

                else:     
            
                    if step % steps_cache_refresh == 0:
                    
                        release infinite_pair_loader, x0 and xT
                        empty cuda cache

                        for i in range(cache_npair // cache_limit)
                        
                            xT = infinite_target_dataloader.get_batch_and_proceed
                            x0 = start from xT sample with the backward_net with no gradient
                            store x0 and xT in the cpu
                        
                        couple accumulated x0 and xT                    
                        
                        infinite_pair_loader = build_infinite_dataloader(batch_size=batch_size, data=coupling)
                    
                    batch = next(infinite_pair_loader)
                    update forward_net using the batch

                step += 1
"""