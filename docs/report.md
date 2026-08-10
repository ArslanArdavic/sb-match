## Overview 

Reference method: Diffusion Schrödinger Bridge Matching (Shi et al., 2023)

Dataset: AFHQ (Animal Faces-HQ)

Source Distr.: AFHQ - Cat

Target Distr.: AFHQ - Wild

Objective: Train diffusion Schrödinger bridge between two distributions.

## Setup 1 - Reference process on local machine
Objective:
1. Sample a state of the Brownian Motion starting from a given instance from the source distribution.
2. Sample an intermediate state of the Brownian Bridge from the given tuple (target, source). 
3. Sample the markovian drift target from the given tuple (target, source).  

Execution: `python3 -m tests.reference`

Results:
1. tests/outputs/reference/brownian_states_sigma_1_parallel.png
2. tests/outputs/reference/conditioned_brownian_bridge_sigma_1_parallel
3. tests/outputs/reference/markovian_drift_target_brownian_bridge_sigma_1.png (Sample the drift and take an EM step, continue with sampling another drift from the resulting state.)


## Setup 2.1 - Forward only on local machine

Objective: 
1. Train the forward bridge for 64x64 images and sample trajectory of one image, on the local machine.
2. Train the forward bridge for 512x512 images and sample trajectory of one image, on the local machine.

Execution:
1. `python3 -m tests.run_forward_only local 64`
2. `python3 -m tests.run_forward_only local 512`

Results:
1. Success
    - 1.1. tests/outputs/forward_only/local/forward_only_64_epoch_losses.png
    - 1.2. tests/outputs/forward_only/local/forward_only_64_net.pt
    - 1.3. tests/outputs/forward_only/local/forward_only_64_step_losses.png
    - 1.4. tests/outputs/forward_only/local/forward_only_sample_trajectory_test_64.png
2. `torch.OutOfMemoryError: CUDA out of memory` even when `batch_size=1`


## Setup 2.2 - Forward only on LUMI

Objective: 
1. Train the forward bridge for 64x64 images and sample trajectory of one image, on LUMI.
2. Train the forward bridge for 512x512 images and sample trajectory of one image, on LUMI.

Execution:
1. `sbatch scripts/test_forward_only_64.sh`
2. `sbatch scripts/test_forward_only_512.sh`

Results:
1. Completed in 00:03:35
    - 1.1. tests/outputs/forward_only/lumi/forward_only_64_epoch_losses.png
    - 1.2. tests/outputs/forward_only/lumi/forward_only_64_step_losses.png
    - 1.3. tests/outputs/forward_only/lumi/forward_only_sample_trajectory_test_64.png
2. Completed in ~ 01:10:00
    - 2.1. tests/outputs/forward_only/lumi/forward_only_sample_trajectory_test_512.png

Notes:
2. Timeout after 30 mins.


## Setup 3 - Forward-Backward Prototype 64x64 on LUMI

Objective:
- Smoke train the bridge for 64x64 images and sample forward trajectory of one image, on LUMI.

Execution:
- `sbatch scripts/forward_backward_prototype.sh`

Configs:
1. {
    "batch_size": 256 ,
    "sample_batch_size": 512,  
    "N" : 8,        
    "n_outer" : 2, 
    "epochs_per_drift": 1,
    }

Results:
1. Completed in 00:10:58
    - 1.1 tests/outputs/prototype/lumi/prototype_backward_64_hpc_epoch_losses.png                             
    - 1.2 tests/outputs/prototype/lumi/prototype_backward_64_hpc_step_losses.png  
    - 1.3 tests/outputs/prototype/lumi/prototype_forward_64_hpc_epoch_losses.png  
    - 1.4 tests/outputs/prototype/lumi/prototype_forward_64_hpc_step_losses.png       
    - 1.5 tests/outputs/prototype/lumi/prototype_forward_sample_trajectory_test_64.png      

## Setup 4 - Forward-Backward Prototype + EMA 64x64 on LUMI

Objective: Averaging of the models with EMA, train and sample for 64x64 images on LUMI

Observation: DDBSM does not implement EMA! 

Conclusion: **SKIP THIS SETUP**

## Setup 5 - Bridge Prototype with Caching 512x512

Objective: Introduce the caching mechanism to prevent sampling over the whole source/target datasets.

Reference: Shi et al. (2023) adapts trajectory caching procedure of De Bortoli et al. (2021)

Observation: 
- DDBSM does not implement caching! 
- DBSM iterates training loop over a constant regarding the number of steps. 

### Setup 5.1 - Smoke Train

Objective: Smoke train. (`bridge_prototype.py`)

Execution: 
- `sbatch scripts/bridge_smoke_512.sh`

Results: 
- Completed in ~7 minutes
    - /flash/project_465002822/sb-match/outputs/bridge/hydra/2026-08-10_16-59-42/

### Setup 5.2 - Full Experiment

Objective: Train models with the setting reported in the (Shi et al., 2023).

Execution:
- `sbatch scripts/bridge_512.sh`

Conclusion: **CANCEL EXPERIMENT**
- Training only the first direction will take more than 16 hours.
