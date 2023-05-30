import sys

import pyrallis
from diffusers.utils import is_wandb_available
from loguru import logger

from training.config import RunConfig


class CoachLogger:

    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.step = 0
        self.configure_loguru()
        self.log_config()
        self.validate_wandb()

    def configure_loguru(self):
        logger.remove()
        format = '<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{message}</level>'
        logger.add(sys.stdout, colorize=True, format=format)
        logger.add(self.cfg.log.logging_dir / 'log.txt', colorize=False, format=format)

    def log_config(self):
        with (self.cfg.log.exp_dir / 'config.yaml').open('w') as f:
            pyrallis.dump(self.cfg, f)
        self.log_message('\n' + pyrallis.dump(self.cfg))

    def validate_wandb(self):
        if self.cfg.log.report_to == "wandb":
            if not is_wandb_available():
                raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    @staticmethod
    def log_message(msg: str):
        logger.info(msg)

    def log_start_of_training(self, total_batch_size: int, num_samples: int):
        self.log_message("***** Running training *****")
        self.log_message(f"  Num examples = {num_samples}")
        self.log_message(f"  Instantaneous batch size per device = {self.cfg.optim.train_batch_size}")
        self.log_message(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        self.log_message(f"  Gradient Accumulation steps = {self.cfg.optim.gradient_accumulation_steps}")
        self.log_message(f"  Total optimization steps = {self.cfg.optim.max_train_steps}")

    def update_step(self, step: int):
        self.step = step
