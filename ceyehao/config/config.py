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
    return EasyDict({
        'mode': 'train',
        'model': 'CEyeNet',
        'model_checkpoint': '../log/CEyeNet/CEyeNet',
        'profile_size': [200, 200],
        'simg_res': [200, 200],
        'device': 'cuda',
        'amp': True,
        'model_cfg': {
            'unet_cfg': {
                'base_num_features': 64,
                'num_pool': 4,
                'down_kwargs': {
                    'block_type': 'conv',
                    'num_conv': 2,
                    'kernel_size': [3, 3],
                    'dilation': [1, 1],
                },
                'up_kwargs': {
                    'block_type': 'conv',
                    'num_conv': 2,
                    'kernel_size': [3, 3],
                    'dilation': [1, 1],
                },
            },
            'ceye_cfg': {
                'compeye_kwargs': {
                    'pool_type': 'avg',
                    'tp_order': 'tile_first',
                    'tile_factor': 7,
                },
            },
            'unpp_cfg': {
                'input_channels': 1,
                'base_num_features': 64,
                'num_classes': 2,
                'num_pool': 4,
                'profile_size': [200, 200],
            },
        },
        'data_cfg': {
            'tilepool': False,
            'x_axial_flip': False,
            'x_randinv': True,
            'x_rand_axial_translate': True,
            'sym_agmentation': 0,
            'data_root_dir': '../dataset',
            'dataset_size': [9000, 1000],
            'train_bs': 8,
            'valid_bs': 8,
            'workers': 4,
        },
        'pix_acc_cfg': {
            'op_flags': ['round', 'match'],
            'matching_error_thresholds': 0.01,
        },
        'percep_acc_cfg': {
            'include_pix_acc': True,
            'perceptual_weights': [4, 4, 4, 4, 4],
        },
        'train_cfg': {
            'max_epoch': 300,
            'log_dir': '../log',
            'log_interval': 225,
            'lr_init': 0.001,
            'factor': 0.1,
            'milestones': [30, 80],
            'weight_decay': 0.0005,
            'momentum': 0.9,
            'is_warmup': True,
            'warmup_epochs': 1,
            'lr_final': 1.0e-05,
            'lr_warmup_init': 0.0,
            'hist_grad': False,
            'loss_weights': [1, 1, 1, 1],
        },
        'loss_cfg': {
            'components': ['l1', 'l1_grad'],
            'weights': [1, 1],
        },
        'eval_cfg': {
            'result_dir': '',
            'exports': {
                'eval_results': True,
                'cfg': True,
                'model_checkpoint': False,
                'plot': {
                    'tt_pred': False,
                    'tt_label': False,
                },
            },
            'programs': {
                'accuracy': True,
                'time': True,
            },
        },
    })


if __name__ == "__main__":
    main()
