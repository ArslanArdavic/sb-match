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
2. 

Notes:
2. Timeout after 30 mins.
