#!/usr/bin/env python

from ceyehao.config.config import parse_args


def main():
    """Run the training script."""
    args = parse_args()
    if args.cfg_path:
        from ceyehao.utils.io import load_cfg_yml
        cfg = load_cfg_yml(args.cfg_path)
    else:
        print("No cfg is provided. Using built-in default config.")
        from ceyehao.config.config import build_default_cfg
        cfg = build_default_cfg()
    from ceyehao.tools.trainer import Trainer
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
