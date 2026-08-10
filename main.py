"""hydra entry point

    python3 -u main.py                                  # conf/config.yaml defaults
    python3 -u main.py experiment=smoke                 # short end-to-end run, every branch, real 512 net
    python3 -u main.py env=local "res='64'"             # group names that look like ints need quoting
    python3 -u main.py "res='64'" n_outer=2 steps_first_round=50 steps_per_drift=20 N=8 wandb.mode=disabled
    python3 -u main.py --multirun lr=1e-4,3e-5          # two sequential jobs
    python3 -u main.py --cfg job                        # print the composed config and exit
"""
import logging

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from train import bridge_prototype

# hydra's default job_logging puts a console handler and a file handler on the root logger,
# so everything below lands on stdout (the slurm .out) and in ${outdir}/hydra/<ts>/main.log
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    log.info("composed config:\n%s", OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.seed)
    bridge_prototype.train(cfg)


if __name__ == "__main__":
    main()