
"""training a bridge model with the caching mechanism

################################################################################################################################

OUR DESING CHOICES:
    1. ALIGNS
    2. ALIGNS
    3. ALIGNS
    4. EFFECTIVELY ALINGS: drops the old handle and empties the cuda cache after the outer != 0 check
    5. EFFECTIVELY ALINGS: indepented coupling for the first outer iteration on both directions, see (b) below
    6. EFFECTIVELY ALINGS: one N for cache simulation and for evaluation
    7. EFFECTIVELY ALINGS: one training tuple per cached pair per visit
    8. cache lives in cpu memory, gone on restart
    9. no resume, a killed job re-simulates the whole cache
    10. simulate the cache with the live training net
    11. cache_npair and steps_cache_refresh set directly, nothing logged
    12. EFFECTIVELY ALINGS: loss_scale hardcoded, the reference gates it on a flag with a std_trick sibling, see docs/loss-scaling.md
    13. EFFECTIVELY ALINGS: sigma set directly, sigma^2 = 5, see (a) below
    14. EFFECTIVELY ALINGS: no noise term at the last em step, reference parametrizes
    15. backward net reparametrized to its own time, tau = 1 - t
    16. ALIGNS
    17. one file per direction, net + optimizer, overwritten after each drift, last 1 kept


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
    12. loss_scale multiplies pred and target by sigma*sqrt(1-t) forward / sigma*sqrt(t) backward, bounding the blowing tail
    13. sigma derived from the gamma schedule, sigma^2 = sum(gammas) = 5 at afhq
    14. modified euler maruyama for the cache, the final step drops the noise term
    15. both nets share the global t, backward simulates it downwards
    16. first_num_iter separate from num_iter, the bridge matching pretrain gets 100000 vs 5000 per outer at afhq
    17. net, optimizer and ema net saved separately per (direction, outer, step), every gif_stride and at drift end,
        nothing pruned, the suffix the three share is what find_last_ckpt resumes from

PAPER vs REFERENCE IMPLEMENTATION MISMATCHES:
    afhq numbers above come from the paper appendix I.5, not from conf/dataset/afhq_transfer.yaml, which is stale on:
    a. gamma: yaml inherits linspace(0.001, 0.2) from conf/method/dbdsb.yaml, paper says constant stepsizes, sigma^2 = 5 -> gamma = 0.05
    b. first_coupling: yaml inherits ref, paper pretrains both nets with bridge matching -> ind
    c. num_iter / first_num_iter: yaml 25000 / equal, paper 5000 / 100000
    d. n_ipf: yaml 30, paper 20
    e. model: yaml pins none (config.yaml default UNET), paper follows Liu et al. 2023b -> DDPMpp_RF
    f. loss weighting: appendix H derives (1 + sigma^2 t/(1-t))^-1, loss_scale implements sigma^2 (1-t), same tail order, no +1 normalization
    the repo has no afhq run script or README line, so the cli overrides that produced the paper run are simply absent
    cache_npar 400 / cache_refresh_stride 1000 are yaml only, the paper gives cache sizing for celeba but not afhq

################################################################################################################################
"""
import logging
import os
import time

import wandb
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import TensorDataset, DataLoader
from diffusers import UNet2DModel

from utils.process_image import load_afhq_train, infinite_dataloader
from diffusion.reference import sample_conditioned_brownian_bridge, sample_markovian_drift_target_brownian_bridge

# goes to both the slurm .out and ${outdir}/hydra/<ts>/main.log, see main.py
log = logging.getLogger(__name__)

def train(config):

    PRM = ["device", "datadir", "ckptdir", "downsize", "cache_limit", "batch_size", "lr", "n_outer",
           "steps_first_round","steps_per_drift", "steps_cache_refresh", "cache_npair", "N", "sigma", "eps", "log_stride"]
    assert not [prm for prm in PRM if prm not in config]

    device              = config["device"]
    datadir             = config["datadir"]
    ckptdir             = config["ckptdir"]
    downsize            = config["downsize"] 
    cache_limit         = config["cache_limit"] 
    batch_size          = config["batch_size"]
    lr                  = config["lr"]
    n_outer             = config["n_outer"]
    steps_first_round   = config["steps_first_round"]
    steps_per_drift     = config["steps_per_drift"]
    steps_cache_refresh = config["steps_cache_refresh"]
    cache_npair         = config["cache_npair"]
    N                   = config["N"]
    sigma               = config["sigma"]
    eps                 = config["eps"]
    log_stride          = config["log_stride"]

    assert cache_npair % cache_limit == 0
    assert cache_npair % batch_size  == 0


    os.makedirs(ckptdir, exist_ok=True)

    run = setup_wandb(config)
    cache_epochs = batch_size * steps_cache_refresh / cache_npair
    data_epochs  = steps_per_drift * cache_npair / (5153 * steps_cache_refresh)
    run.config.update({"cache_epochs": cache_epochs, "data_epochs": data_epochs})

    log.info(
        # run.name is None until an offline run is synced, run.id exists in both modes
        "run %s | %dpx | %d outer x (%d + %d) steps | bs %d | cache %d pairs / %d chunk / N %d "
        "| cache_epochs %.2f | data_epochs %.4f",
        run.id, downsize, n_outer, steps_first_round, steps_per_drift, batch_size,
        cache_npair, cache_limit, N, cache_epochs, data_epochs,
    )

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

        for direction in ("backward", "forward"):

            if direction == "forward":          

                forward_net.train()

                step = 0

                if outer == 0:
                    step_limit = steps_first_round
                else:
                    step_limit = steps_per_drift

                log.info("outer %d/%d | forward | %d steps", outer, n_outer - 1, step_limit)
                stage_start = time.time()

                while step < step_limit:

                    if outer != 0 and step % steps_cache_refresh == 0:

                        infinite_pair_loader, X0_sim, XT = None, None, None
                        torch.cuda.empty_cache()

                        X0_sim = torch.empty(cache_npair, 3, downsize, downsize) 
                        XT     = torch.empty(cache_npair, 3, downsize, downsize) 

                        backward_net.eval()
                        refresh_start = time.time()

                        for i in range(cache_npair // cache_limit):
                            xT, _ = next(infinite_target_cache_dataloader) # dismiss the label
                            
                            # Sample
                            with torch.no_grad():
                                xt = xT.to(device)
                                for j in range(N):
                                    t = torch.full((xt.shape[0],), j / N, device=device)
                                    vt = backward_net(xt, t * 1000).sample 
                                    if j == N-1:
                                        xt = xt + vt * dt 
                                    else: 
                                        xt = xt + vt * dt + sigma * torch.sqrt(dt) * torch.randn_like(xt)

                            X0_sim[i*cache_limit : (i+1)*cache_limit] = xt.cpu()
                            XT[i*cache_limit : (i+1)*cache_limit]     = xT

                        infinite_pair_loader =  infinite_dataloader(DataLoader(TensorDataset(X0_sim, XT), batch_size=batch_size, shuffle=True, drop_last=True))

                        log.info("  cache refresh at step %d | %d pairs simulated backward in %.1fs",
                                 step, cache_npair, time.time() - refresh_start)

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
                    w = (sigma * torch.sqrt(1 - t)).view(-1, 1, 1, 1) # loss_scale, bounds the t -> 1 tail
                    loss = criterion(w * pred, w * target)
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(forward_net.parameters(), 1.0) # pre-clip gradient norm
                    forward_optimizer.step()
                    forward_optimizer.zero_grad()

                    if step % log_stride == 0 :
                        # no step= kwarg: wandb's commit counter advances on its own, global_step is the x axis
                        run.log({
                            "forward/loss"      : loss.item(),
                            "forward/grad_norm" : grad_norm.item(),
                            "global_step"       : compute_global_step(outer, step, steps_first_round, steps_per_drift),
                        })
                        log.info("outer %d | forward | step %5d/%d | loss %.4f | grad_norm %.3f",
                                 outer, step, step_limit, loss.item(), grad_norm.item())

                    step += 1

                log.info("outer %d/%d | forward | done in %.1fs", outer, n_outer - 1, time.time() - stage_start)

                save_drift(forward_net, forward_optimizer, "forward", outer,
                           compute_global_step(outer, step, steps_first_round, steps_per_drift),
                           downsize, ckptdir, run)

            elif direction == "backward":     

                backward_net.train()

                step = 0

                if outer == 0:
                    step_limit = steps_first_round
                else:
                    step_limit = steps_per_drift

                log.info("outer %d/%d | backward | %d steps", outer, n_outer - 1, step_limit)
                stage_start = time.time()

                while step < step_limit:

                    if outer != 0 and step % steps_cache_refresh == 0:

                        infinite_pair_loader, XT_sim, X0 = None, None, None
                        torch.cuda.empty_cache()

                        X0      = torch.empty(cache_npair, 3, downsize, downsize) 
                        XT_sim  = torch.empty(cache_npair, 3, downsize, downsize) 

                        forward_net.eval()
                        refresh_start = time.time()

                        for i in range(cache_npair // cache_limit):
                            x0, _ = next(infinite_source_cache_dataloader) # dismiss the label
                            
                            # Sample
                            with torch.no_grad():
                                xt = x0.to(device)
                                for j in range(N):
                                    t = torch.full((xt.shape[0],), j / N, device=device)
                                    vt = forward_net(xt, t * 1000).sample 
                                    if j == N-1:
                                        xt = xt + vt * dt 
                                    else:
                                        xt = xt + vt * dt + sigma * torch.sqrt(dt) * torch.randn_like(xt)

                            X0[i*cache_limit : (i+1)*cache_limit]       = x0
                            XT_sim[i*cache_limit : (i+1)*cache_limit]   = xt.cpu()

                        infinite_pair_loader =  infinite_dataloader(DataLoader(TensorDataset(X0, XT_sim), batch_size=batch_size, shuffle=True, drop_last=True))

                        log.info("  cache refresh at step %d | %d pairs simulated forward in %.1fs",
                                 step, cache_npair, time.time() - refresh_start)

                    if outer == 0:
                        x0, _ = next(infinite_source_train_dataloader)
                        xT, _ = next(infinite_target_train_dataloader)
                    else:
                        x0, xT = next(infinite_pair_loader)

                    # Update backward_net 
                    x0, xT  = x0.to(device), xT.to(device)
                    t = torch.rand(x0.shape[0], device=device) * (1 - 2*eps) + eps
                    xt = sample_conditioned_brownian_bridge(x0=xT, xT=x0, sigma=sigma, t=t)
                    target = sample_markovian_drift_target_brownian_bridge(xt, xT=x0, t=t)

                    pred = backward_net(xt, t * 1000).sample
                    w = (sigma * torch.sqrt(1 - t)).view(-1, 1, 1, 1) # loss_scale, sigma*sqrt(t) in global time, tau = 1 - t here
                    loss = criterion(w * pred, w * target)
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(backward_net.parameters(), 1.0) # pre-clip gradient norm
                    backward_optimizer.step()
                    backward_optimizer.zero_grad()

                    if step % log_stride == 0 :
                        # no step= kwarg: wandb's commit counter advances on its own, global_step is the x axis
                        run.log({
                            "backward/loss"      : loss.item(),
                            "backward/grad_norm" : grad_norm.item(),
                            "global_step"        : compute_global_step(outer, step, steps_first_round, steps_per_drift),
                        })
                        log.info("outer %d | backward | step %5d/%d | loss %.4f | grad_norm %.3f",
                                 outer, step, step_limit, loss.item(), grad_norm.item())

                    step += 1

                log.info("outer %d/%d | backward | done in %.1fs", outer, n_outer - 1, time.time() - stage_start)

                save_drift(backward_net, backward_optimizer, "backward", outer,
                           compute_global_step(outer, step, steps_first_round, steps_per_drift),
                           downsize, ckptdir, run)

    log.info("training finished, %d outer iterations", n_outer)
    run.finish()

def compute_global_step(outer, step, steps_first_round, steps_per_drift):
    # both directions of an outer share a range, so each wandb series comes out continuous
    if outer == 0:
        return step
    return steps_first_round + (outer - 1) * steps_per_drift + step

def save_drift(net, optimizer, direction, outer, global_step, downsize, ckptdir, run):
    """one file per direction, overwritten after every drift. keeps the last drift only."""

    path = os.path.join(ckptdir, f"drift_{direction}.pt")
    tmp  = path + ".tmp"   # same directory, so os.replace below stays atomic

    torch.save({
        "net"         : net.state_dict(),
        "optimizer"   : optimizer.state_dict(),
        "direction"   : direction,
        "outer"       : outer,
        "global_step" : global_step,
        "downsize"    : downsize,   # build_models(downsize) rebuilds the shell this loads into
    }, tmp)
    os.replace(tmp, path)

    # summary, not log: one current value per direction, matching the one file we keep
    run.summary[f"{direction}/ckpt_outer"] = outer
    run.summary[f"{direction}/ckpt_path"]  = path

    log.info("outer %d/%s | saved %s", outer, direction, path)


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

    # hydra hands us a DictConfig; wandb wants a plain dict, resolved (${...} expanded)
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)

    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity=config["wandb"]["entity"],
        # Set the wandb project where this run will be logged.
        project=config["wandb"]["project"],
        mode=config["wandb"]["mode"],
        # Track hyperparameters and run metadata.
        config=config,
    )

    # Plot everything against global_step instead of wandb's own commit counter. Both directions of an
    # outer share a global_step range, so each series comes out continuous rather than a 50% duty cycle comb.
    run.define_metric("global_step")
    run.define_metric("*", step_metric="global_step")

    return run
