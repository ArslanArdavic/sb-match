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

Entrypoint: tests/reference.py 

Results:
1. tests/outputs/reference/brownian_states_sigma_1_parallel.png
2. tests/outputs/reference/conditioned_brownian_bridge_sigma_1_parallel
3. tests/outputs/reference/markovian_drift_target_brownian_bridge_sigma_1.png (Sample the drift and take an EM step, continue with sampling another drift from the resulting state.)


## Setup 2 - 