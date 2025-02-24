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
        
    args = parser.parse_args(str_args)
    return args
