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


def parse_args(str_args: Optional[list] = None):
    parser = argparse.ArgumentParser(description="configure model from command line")
    parser.add_argument(
        "--cfg_path",
        type=str,
        default="",
        help="path to the config yaml file",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="",
        help="Inverse Design: path to the root log dir for search outputs",
    )
    parser.add_argument(
        "--tgt_pth",
        type=str,
        default="",
        help="Inverse Design: path to the target flow profile image",
    )

    parser.add_argument(
        "--num_chn",
        type=int,
        default=10000,
        help="Inverse Design (random search only): number of microchannels to screen",
    )

    parser.add_argument(
        "--num_hao",
        type=int,
        default=8,
        help="Inverse Design (random search only): number of haos in each microchannel ",
    )
    parser.add_argument(
        "--num_pin",
        type=int,
        default=5,
        help="Inverse Design (random search only): number of input flow porfiles for each microchannel ",
    )
    parser.add_argument(
        "--max_gen",
        type=int,
        default=1000,
        help="Inverse Design (GA search only): number of generations",
    )
    parser.add_argument(
        "--max_num_hao",
        type=int,
        default=8,
        help="Inverse Design (GA search only): max number of haos in each microchannel ",
    )
    parser.add_argument(
        "--min_num_hao",
        type=int,
        default=4,
        help="Inverse Design (GA search only): min number of haos in each microchannel ",
    )
    parser.add_argument(
        "--elitism",
        type=float,
        default=0.03,
        help="Inverse Design (GA search only): elitism ratio",
    )
    parser.add_argument(
        "--mutation",
        type=float,
        default=0.35,
        help="Inverse Design (GA search only): elitism ratio",
    )
    parser.add_argument(
        "--prob_select",
        type=bool,
        default=False,
        help="Inverse Design (GA search only): enable probablistic selection of parents",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=100,
        help="Inverse Design (GA search only): population",
    )
    parser.add_argument(
        "--restrict_flow_ratio",
        type=bool,
        default=True,
        help="Inverse Design (GA search only): restrict the flow_ratio range in sampling input flow profile",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default='IoU',
        help="Inverse Design (GA search only): metrics to evaluate output flow profile ",
    )

    args = parser.parse_args(str_args)
    return args
