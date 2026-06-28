
## forward_backward_prototype (64px)

```
Possibly will TIMEOUT due 6 hours limit.
Submitted same job with JobID: 
  - 24 hours limit
  - "sample_batch_size"=1024
```

JobID: 19576722
log:   pfs/lustrep3/projappl/project_465002822/sb-match/tests/outputs/log/forward_backward_prototype_19576722.out

config = {
      "device"    : "cuda",
      "datadir"   : "/flash/project_465002822/sb-match/data/afhq/train/",  
      "downsize"  : 64 ,
      "batch_size": 256 ,
      "sample_batch_size": 512,
      "gradient_cp": False,
      "lr"    : 1e-4,
      "N" : 100,
      "n_outer" : 15, 
      "epochs_per_drift": 10,
      "eps"   : 1e-4,
      "sigma" : 1,
    }

outdir : "/flash/project_465002822/sb-match/tests/outputs"

model : "prototype_forward_64_hpc_net.pt"



## forward_only (512px)
JobID: 19563446
elapsed: 22:11:18
log: pfs/lustrep3/projappl/project_465002822/sb-match/tests/outputs/log/forward_only_train_full_size_19563446.out

config = {"device"    : "cuda",
          "datadir"   : "/flash/project_465002822/sb-match/data/afhq/train/",  
          "downsize"  : 512 ,
          "batch_size": 4 ,
          "lr"    : 1e-4,
          "epochs": 20,
          "eps"   : 1e-4,
          "sigma" : 1,
        }

outdir : "/flash/project_465002822/sb-match/tests/outputs"

model  : "forward_only_512_net.pt"



## forward_only (64px)
JobID: -
log: - 

config = {"device"    : "cuda",
          "datadir"   : "/home/arslan/research/literature/foundations-schrodinger-bridges-tang-2026/sb-match/data/afhq/train/",  
          "downsize"  : 64 ,
          "batch_size": 8 ,
          "lr"    : 1e-4,
          "epochs": 10,
          "eps"   : 1e-4,
          "sigma" : 1,
        }

outdir : "/tests/outputs"

model  : "forward_only_64_net.pt" 