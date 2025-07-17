import os
import pickle
from easydict import EasyDict
import re
import yaml
import copy
import numpy as np

from ceyehao.utils.utils import mkdirs


def pickle_load_config(infer_weight_path):
    cfg = EasyDict()
    model_dir = os.path.dirname(infer_weight_path)
    if os.path.exists(os.path.join(model_dir, "app_cfg.pickle")):
        with open(os.path.join(model_dir, "app_cfg.pickle"), "rb") as f:
            original_cfg = pickle.load(f)
    elif os.path.exists(os.path.join(model_dir, "cfg.pickle")):
        with open(os.path.join(model_dir, "cfg.pickle"), "rb") as f:
            original_cfg = pickle.load(f)
    else:
        raise FileNotFoundError(
            "Could not find the model config file. shoud be app_cfg.pickle or cfg.pickle"
        )

    cfg.__build_default__("infer", original_cfg.model, original_cfg.profile_size)
    cfg.model_cfg = original_cfg.model_cfg
    cfg.data_cfg.compeye = original_cfg.data_cfg.compeye
    cfg.eval_cfg.infer_weight_path = infer_weight_path
    if hasattr(cfg.model_cfg, "block_type") and cfg.model_cfg.block_type == "res":
        cfg.model_cfg.dilation = [
            1,
        ] * cfg.model_cfg.num_conv
        cfg.model_cfg.kernel_size = [
            3,
        ] * cfg.model_cfg.num_conv
    return cfg

def load_cfg_yml(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
        cfg = EasyDict(cfg)
    return cfg

def convert_cfg_to_dict(cfg):
    """convert the config object or easydict object to native dict."""
    if isinstance(cfg, EasyDict):
        cfg = cfg.__dict__
    for k, v in cfg.items():
        if isinstance(v, EasyDict):
            cfg[k] = convert_cfg_to_dict(v)
    return cfg

def dump_cfg_yml(cfg, cfg_path):
    with open(cfg_path, "w") as f:
        cfg = convert_cfg_to_dict(copy.deepcopy(cfg))
        yaml.dump(cfg, f)

def mirror_model_dir_tree(dest_dir, model_path):
    """mirror the model directory structure in the result directory."""
    # extract each level of directory
    dirs = os.path.relpath(model_path, r"../").split(os.sep)
    save_path = os.path.join(dest_dir, *dirs[1:-1])
    return mkdirs(save_path, remove_old=False)
    # load model predictor.


def find_index(a):
    index = re.findall("(?<=-)\d+", a)
    if index:
        return int(index[0])
    else:
        return None


class FindIndex:
    def __init__(self, pattern="(?<=-)\d+") -> None:
        self.pattern = pattern

    def __call__(self, a):
        index = re.findall(self.pattern, a)
        if index:
            return int(index[0])
        else:
            return None

def read_obs_param(f):
    """read a single obstacle data from a list of file."""
    first_line = f.readline()
    if not first_line:
        return None, None, None
    obs_no = int(first_line.removeprefix("# Obstacle ").removesuffix("\n"))
    coord_str = ""
    for _ in range(6): 
        coord_str += f.readline().removesuffix("\n") + "\t"
    coord = np.fromstring(coord_str, sep="\t") / 1e6
    pos_str = f.readline().removeprefix("# ").removesuffix("\n")
    pos = True if pos_str == "Top" else False
    return obs_no, coord, pos

def write_obs_param(f, obs_no, coord, pos):
    """write a single obstacle data to file.
    obs_no start from 1"""
    assert pos in [True, False, "Top", "Bottom"], "pos is not valid."
    pos_str = "Top" if pos in ["Top", True] else "Bottom"
    f.write(f"# Obstacle {obs_no}\n")
    for rows in coord.reshape(6, 3) * 1e6:
        for ele in rows:
            f.write(f"{ele:6.1f}\t")
        f.write("\n")
    f.write(f"# {pos_str}\n")

def read_obs_param_list(f):
    """read a list of obstacle data from a file."""
    obs_nos = []
    coords = []
    poses = []
    while True:
        obs_no, coord, pos = read_obs_param(f)
        if obs_no is None:
            break
        obs_nos.append(obs_no)
        coords.append(coord)
        poses.append(pos)
    return np.array(obs_nos), np.array(coords), np.array(poses)

def write_obs_param_list(f, obs_nos, coords, poses):
    """write a list of obstacle data to a file."""
    for obs_no, coord, pos in zip(obs_nos, coords, poses):
        write_obs_param(f, obs_no, coord, pos)

