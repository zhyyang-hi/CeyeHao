#!/usr/bin/env python

from config.config import parse_args
from tools.trainer import Trainer
from utils.io import load_cfg_yml


if __name__ == "__main__":
    args = parse_args()
    cfg = load_cfg_yml(args.cfg_path)
    trainer = Trainer(cfg)
    trainer.train()
