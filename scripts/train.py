import sys

import pyrallis
from diffusers.utils import check_min_version

sys.path.append(".")
sys.path.append("..")

from training.coach import Coach
from training.config import RunConfig

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.14.0")


@pyrallis.wrap()
def main(cfg: RunConfig):
    prepare_directories(cfg=cfg)
    coach = Coach(cfg)
    coach.train()


def prepare_directories(cfg: RunConfig):
    cfg.log.exp_dir = cfg.log.exp_dir / cfg.log.exp_name
    cfg.log.exp_dir.mkdir(parents=True, exist_ok=True)
    cfg.log.logging_dir = cfg.log.exp_dir / cfg.log.logging_dir
    cfg.log.logging_dir.mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    main()
