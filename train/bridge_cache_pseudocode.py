"""
TODO: COPY THE UPDATED VERSION HERE
"""

"""pseudocode for training a bridge without caching mechanism

build_dataloaders
    
store all source and target images in memory (X0, XT)

create a coupling dataset from X0 and XT

forward_net , backward_net = build_models

define loss function and build_optimizers

for outer in range(n_outer):

    for direction in ("forward", "backward"):
    
        if direction == "forward":
        
            for epoch in range(epochs_per_drift):
            
                for x0, xT in DataLoader(coupling):
                
                    update forward_net using x0, xT
            
            start from XO sample with the forward_net and store as the new coupling

        elif direction == "backward":
        
            for epoch in range(epochs_per_drift):
                            
                for x0, xT in DataLoader(coupling):
                
                    update backward_net using x0, xT

                    
            start from XT sample with the backward_net and store as the new coupling        
"""


"""pseudocode for training a bridge model with the caching mechanism

################################################################################################################################

OUR DESING CHOICES:
    1. ALIGNS
    2. ALIGNS
    3. ALIGNS
    4. ALIGNS
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
    

infinite_source_dataloader = build_infinite_dataloader(batch_size=cache_limit, data=source)
infinite_target_dataloader = build_infinite_dataloader(batch_size=cache_limit, data=target)

infinite_source_train_dataloader = build_infinite_dataloader(batch_size=batch_size, data=source)
infinite_target_train_dataloader = build_infinite_dataloader(batch_size=batch_size, data=target)

forward_net , backward_net = build_models
define loss function and build_optimizers


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


##################### REFERENCE STRUCTURE with OUR DESIGN CHOICES #####################                 

for outer in range(n_outer):

    for direction in ("bacward", "forward"):

        if direction == "forward":        
        
            step = 0

            while step < steps_per_drift:
            
                if step % steps_cache_refresh == 0:
                                    
                    release infinite_pair_loader, x0 and xT
                    empty cuda cache

                    if outer != 0:

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

"""