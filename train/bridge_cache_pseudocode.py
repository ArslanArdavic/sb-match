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
TODO: backward branch

infinite_source_dataloader = build_infinite_dataloader(batch_size=cache_npair, data=source)
infinite_target_dataloader = build_infinite_dataloader(batch_size=cache_npair, data=target)

forward_net , backward_net = build_models
define loss function and build_optimizers


for outer in range(n_outer):
    for direction in ("forward", "backward"):

        if direction == "forward":

            
            step = 0
            while step < steps_per_drift:
                if step % steps_cache_refresh == 0:
                    xT = infinite_target_dataloader.get_batch_and_proceed
                    if outer == 0:
                        x0 = infinite_source_dataloader.get_batch_and_proceed
                    else:
                        for i in range(0, cache_npair // cache_limit)
                            start from xT[i*cache_limit:(i+1)*cache_limit] sample with the backward_net and store in the cpu 
                        x0 is the collection of generated sampled
                    couple xT and x0                    
                    infinite_pair_loader = build_infinite_dataloader(batch_size=batch_size, data=coupling)
                
                batch = next(infinite_pair_loader)
                update forward_net using the batch

                step += 1
"""