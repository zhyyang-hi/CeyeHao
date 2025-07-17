from json import load
from typing import Optional, Union
from easydict import EasyDict
import torch
import pickle
import os
import argparse
from argparse import Namespace

SUPPORTED_MODELS = [
    "UNet",
    "AsymUNet",
    "UNet++",
    "CEyeNet",
    "gvtn",
]

MODE = ["train", "eval", "infer"]


def list_config(cfg: EasyDict, prefix=""):
    str_cfg = ""
    for k, v in cfg.items():
        if isinstance(v, EasyDict):
            sub_prefix = prefix + " | " + str(k)
            str_cfg += list_config(v, sub_prefix)
        # skip functions
        elif not callable(v):
            str_cfg += f"{prefix} | {k}: {v}\n"
    str_cfg += "-----\n"
    return str_cfg


def build_argparser():
    parser = argparse.ArgumentParser(
        description="CeyeHao: Unified CLI for Training, GUI, and Search. Use the appropriate arguments for your task.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # GUI group
    gui_group = parser.add_argument_group('GUI Arguments', 'Arguments used for launching the GUI')
    gui_group.add_argument(
        "--cfg_path",
        type=str,
        default="",
        help="Path to the config YAML file for the GUI or other tasks."
    )

    # Training group
    train_group = parser.add_argument_group('Training Arguments', 'Arguments used for model training')
    # (Add more training-specific CLI args here if needed)
    # Currently, training is mostly configured via YAML, so only cfg_path is relevant.

    # Search group
    search_group = parser.add_argument_group('Search Arguments', 'Arguments used for inverse design/search')
    search_group.add_argument(
        "--search_mode",
        type=str,
        choices=["random", "ga"],
        default="random",
        help="Search mode: 'random' for InvDesign, 'ga' for genetic algorithm search."
    )
    search_group.add_argument(
        "--log_dir",
        type=str,
        default="",
        help="Path to the root log directory for search outputs."
    )
    search_group.add_argument(
        "--tgt_pth",
        type=str,
        default="",
        help="Path to the target flow profile image."
    )
    search_group.add_argument(
        "--num_chn",
        type=int,
        default=10000,
        help="(Random search only) Number of microchannels to screen."
    )
    search_group.add_argument(
        "--num_hao",
        type=int,
        default=8,
        help="(Random search only) Number of haos in each microchannel."
    )
    search_group.add_argument(
        "--num_pin",
        type=int,
        default=5,
        help="(Random search only) Number of input flow profiles for each microchannel."
    )
    search_group.add_argument(
        "--max_gen",
        type=int,
        default=1000,
        help="(GA search only) Number of generations."
    )
    search_group.add_argument(
        "--max_num_hao",
        type=int,
        default=8,
        help="(GA search only) Max number of haos in each microchannel."
    )
    search_group.add_argument(
        "--min_num_hao",
        type=int,
        default=4,
        help="(GA search only) Min number of haos in each microchannel."
    )
    search_group.add_argument(
        "--elitism",
        type=float,
        default=0.03,
        help="(GA search only) Elitism ratio."
    )
    search_group.add_argument(
        "--mutation",
        type=float,
        default=0.35,
        help="(GA search only) Mutation ratio."
    )
    search_group.add_argument(
        "--prob_select",
        action="store_true",
        help="(GA search only) Enable probabilistic selection of parents."
    )
    search_group.add_argument(
        "--population",
        type=int,
        default=100,
        help="(GA search only) Population size."
    )
    search_group.add_argument(
        "--restrict_flow_ratio",
        action="store_true",
        help="(GA search only) Restrict the flow_ratio range in sampling input flow profile."
    )
    search_group.add_argument(
        "--metric",
        type=str,
        default='IoU',
        help="(GA search only) Metric to evaluate output flow profile."
    )
    return parser


def parse_args(str_args: Optional[list] = None):
    parser = build_argparser()
    args = parser.parse_args(str_args)
    return args


def main():
    parser = build_argparser()
    args = parser.parse_args()
    print("Parsed arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")


def build_default_cfg():
    """
    Build a default config instance (EasyDict) with the same structure and values as template.yml.
    """
    from ceyehao.utils.io import load_cfg_yml
    cfg = load_cfg_yml('ceyehao/config/template.yml')
    return cfg


if __name__ == "__main__":
    main()
