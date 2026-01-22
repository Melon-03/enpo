#!/usr/bin/env python

import os
# Set tokenizers parallelism to avoid warnings when forking processes
# This must be set before any tokenizer is imported or used
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import hydra
from project.launch_utils.exceptions import print_exceptions
from omegaconf import DictConfig

@hydra.main(config_path="config", config_name="train", version_base=None)
@print_exceptions
def main(config: DictConfig):
    # Only import after hydra has been initialized so that it can be run on login node to set up slurm jobs
    from train import train
    train(config)

if __name__ == "__main__":
    main()
