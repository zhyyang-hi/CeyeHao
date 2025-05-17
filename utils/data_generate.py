import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import torch
import math
import pickle
from tqdm import tqdm
import os

from utils.io import find_index


class HAOParamSampler:
    """generate the coordinates for HAOs. Assume microchannel width is 1."""

    def __init__(
        self,
        num_seg=5,
        xi0_half_interval=0.000175 / 0.0003,
        mean_delta_x=0.00015 / 0.0003,
        var_delta_x=0.4 * 0.00015 / 0.0003,
        yi_a=10,
        yi_b=4,
        # TODO, rewrite by define sw_min, sw_max.
    ) -> None:
        self.param = None
        self.coord_meta = {
            "num_seg": num_seg,
            "xi0_half_interval": xi0_half_interval,
            "mean_delta_x": mean_delta_x,
            "var_delta_x": var_delta_x,
            "yi_a": yi_a,
            "yi_b": yi_b,
        }

    @staticmethod
    def compute_yi(param: np.ndarray, a=10, b=4):
        """
        scale y params by y*(a-num_seg)+b and then convert to weights"""
        modified_param = param.copy()
        ratio_cols = modified_param[..., 5::3]
        num_seg = ratio_cols.shape[-1]
        ratio_cols = ratio_cols * (a - num_seg) + b
        sw = ratio_cols / ratio_cols.sum(-1, keepdims=True)

        y_coords = np.cumsum(sw, axis=-1)

        modified_param[..., 5::3] = y_coords.round(6)
        modified_param[..., 2] = 0
        return modified_param

    # TODO check compute yi is bug free

    def sample_coords(self, num_obs, quasi_rand=True):
        """generate a tensor of parameter list for  obstacle simulation.
        for each obstacle the parameters are: [xn0, xn1, yn for n in linspace(0,5)]

        """
        pass
        param = self.sample_param(num_obs, self.coord_meta["num_seg"], quasi_rand)
        coords = self.param2coord(param)
        self.param = param
        self.coords = coords
        return coords

    @staticmethod
    def sample_param(num_obs, num_seg, quasi_rand=True):
        num_param = (num_seg + 1) * 3

        if quasi_rand:
            soboleng = torch.quasirandom.SobolEngine(dimension=num_param)
            param = soboleng.draw(num_obs)
        else:
            param = torch.rand((num_obs, num_param))
        param = param.detach().numpy()
        return param

    @staticmethod
    def param2coord(param, metadata):
        """param of shape (..., n)"""
        coords = HAOParamSampler.compute_yi(param, metadata["yi_a"], metadata["yi_b"])
        # scale x0 to [-xi0_range, xi0_range]
        coords[..., 0::3] = (2 * coords[..., 0::3] - 1) * metadata["xi0_half_interval"]

        # compute x1= x0 + delta_x, delta_x = mean_delta_x + var_delta_x * rand
        coords[..., 1::3] = (
            (2 * coords[..., 1::3] - 1) * metadata["var_delta_x"]
            + metadata["mean_delta_x"]
            + coords[..., 0::3]
        )
        return coords

    @staticmethod
    def get_widths_from_coords(coords: np.ndarray):
        """coords of shape (..., 3*(num_seg+1))"""
        y = coords[..., 2::3]
        width = y[..., 1:] - y[..., :-1]
        return width

    def get_widths_from_params(self, params: np.ndarray):
        """coords of shape (..., 3*(num_seg+1))"""
        coords = self.param2coord(params)
        y = coords[..., 2::3]
        width = y[..., 1:] - y[..., :-1]
        return width

    def save(self, fname):
        with open(fname, "wb") as pickle_out:
            pickle.dump(self, pickle_out)

    def read_param(self, param_dir):
        param = np.loadtxt(param_dir, delimiter=",")
        self.param = torch.tensor(param)
        print(f"loaded param with shape {param.shape} ")

    def wrt_param(self, fname):
        np.savetxt(
            fname + "_COMSOL.txt",
            self.param,
            fmt="%.7f",
            delimiter=",",
            newline="},{",
            header="{{",
            footer="}}",
        )
        np.savetxt(
            fname + "_np.txt", self.param, fmt="%.7f", delimiter=",", newline="\n"
        )
        print(f"File saved as '{fname}_COMSOL.txt' and '{fname}_np.txt'")


def stl_start_grid(
    y_width: float,
    z_height: float,
    ny: int,
    nz: int,
    fname="default stl start",
    unit="mm",
):
    """Generate coordinates of streamline starting points for FE simulation postprocess
    poins in the form of x and y list. save to  text files.

    Args:
        y_width (float): width of the microchannel
        z_height (float): height of the microchannel
        ny (int): number of pixels in y
        nz (int): number of puxels in z
        fname (str, optional): file name to save the list. Defaults to "default stl start".

    Returns:
        list of y and z
    """
    y_margin = y_width / ny / 2
    y_end = y_width - y_margin
    y = np.linspace(y_margin, y_end, ny)

    z_margin = z_height / nz / 2
    z_end = z_height - z_margin
    z = np.linspace(z_margin, z_end, nz)

    yv, zv = np.meshgrid(y, z)
    yv, zv = yv.flatten(), zv.flatten()

    if unit == "m":
        pass
    elif unit == "mm":
        yv = yv * 1000
        zv = zv * 1000
    else:
        ValueError("wrong unit! only m or mm is supported.")

    np.savetxt(fname + "-y.txt", yv, fmt="%4.3e", delimiter=",", newline=" ")
    np.savetxt(fname + "-z.txt", zv, fmt="%4.3e", delimiter=",", newline=" ")
    print(f"start point coordinates saved to {fname}-y.txt and {fname}-z.txt")
    return yv.flatten(), zv.flatten()


class StreamlineDataProcessor:
    def __init__(
        self,
        cw:list,
        res:list,
        raw_sl_dir=None,
        output_dir=None,
    ) -> None:
        self.raw_sl_dir = raw_sl_dir
        self.output_dir = output_dir
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

        self.cw = cw
        self.res = res

    def batch_preprocess(self, raw_dir=None, output_dir=None):
        raw_dir = raw_dir or self.raw_sl_dir
        output_dir = output_dir or self.output_dir
        self.raw_sl_list = os.listdir(raw_dir)
        if find_index(self.raw_sl_list[0]):
            self.raw_sl_list.sort(key=find_index)
        else:
            self.raw_sl_list.sort()

        for sl in tqdm(self.raw_sl_list):
            self.preprocess_single(os.path.join(raw_dir, sl), output_dir)

    def preprocess_single(self, raw_fname, output_dir=None, nan_imputation=True, mr=0):
        output_dir = output_dir or self.output_dir
        sl_df = self.load_stl(raw_fname)
        sl_mat = self.sl2mat(sl_df, self.cw, self.res, nan_imputation, mr)
        if output_dir:
            save_fname = os.path.join(
                output_dir, os.path.basename(raw_fname).replace(".txt", ".npy")
            )
            np.save(save_fname, sl_mat)
            print(f"saved to {save_fname}")
        return sl_mat

    @staticmethod
    def load_stl(fname: str) -> pd.DataFrame:
        raw = pd.read_fwf(
            fname,
            widths=[26, 25, 25, 25, 25, 25, 20],
            usecols=[0, 1, 2, 3, 4, 5],
            skiprows=7,
        )
        raw = raw.round(7)
        sorted_sl = raw.sort_values(by=["y start", "z start"], ignore_index=True)
        return sorted_sl

    @staticmethod
    def sl2mat(pd_sl_list, cw: list, res: list, nan_imputation=True, mr=0):
        w_min = min(cw[0]*mr, 0)
        w_max = max(cw[0]* (1+mr), cw[0])
        h_res = int(res[0]* (1+abs(mr)))
        # assert y start is within the channel width
        assert (
            pd_sl_list["y end"].max() <= w_max  and pd_sl_list["y end"].min() >= w_min
        ), "Y  coord out of channel dimension."
        assert (
              pd_sl_list["z end"].max() <= cw[1] and pd_sl_list["z end"].min() >=0 
        ), "Z  coord out of channel dimension."

        # remove incomplete streamlines: start/end distance below regular.
        ave_chn_len = np.round(
            pd_sl_list["x end"].max() - pd_sl_list["% x start"].min(), 6
        )
        raw_lite = pd_sl_list[
            (pd_sl_list["x end"] - pd_sl_list["% x start"] >= (ave_chn_len - 5e-6))
        ].iloc[:, [1, 2, 4, 5]]

        # list of boundary of each pixel
        bound_0 = np.linspace(w_min, w_max, h_res + 1)
        bound_1 = np.linspace(0, cw[1], res[1] + 1)

        # pixelize only the end points, not the start points.
        sl_list = raw_lite.copy()
        for i in range(bound_0.shape[0]):
            sl_list.loc[raw_lite["y end"] > bound_0[i], "y end"] = i
        for i in range(bound_1.shape[0]):
            sl_list.loc[raw_lite["z end"] > bound_1[i], "z end"] = i
        # pandas dataframe to numpy array
        sl_list = sl_list.to_numpy()  # to numpy

        # rescale start coords to (0 ~ mat_size)
        sl_list[:, 0] = (sl_list[:, 0]-w_min) / ((w_max-w_min) / h_res)
        sl_list[:, 1] = sl_list[:, 1] / (cw[1] / res[1])

        sl_mat = np.zeros((h_res, res[1], 2))
        sl_mat[:, :] = None
        for sl in sl_list:
            sl_mat[int(sl[2]), int(sl[3])] = [sl[0], sl[1]]

        # from inlet coord to displacement
        end_coord = np.indices([h_res, res[1]]).transpose([1, 2, 0]) + 0.5
        displ_mat = sl_mat - end_coord

        # remove NaN
        if nan_imputation == True:
            displ_mat = NanImputation(displ_mat)
        return displ_mat


def NanImputation(tt):
    """impute the nan value from the raw displacement matrix."""
    x = tt.copy()
    it = 0
    nNaN = np.argwhere(np.isnan(x))
    while nNaN.shape[0]:
        print("current iteration:", it + 1)
        nNaN = np.argwhere(np.isnan(x))
        print("total number of missing value:", nNaN.shape[0])
        for i in nNaN:
            i0, i1, i2 = i
            x[i0, i1, i2] = np.nanmean(
                x[max(i0 - 1, 0) : i0 + 2, max(i1 - 1, 0) : i1 + 2, i2]
            )
        print(f"#{it+1} iteration complete")
        it += 1
        nNaN = np.argwhere(np.isnan(x))
        print("remaining number of missing value:", nNaN.shape[0])
    return x


def laplace_sampler(size, miu: float, b: float, rng=None):
    """sampling from laplace distribution
    laplace distribution: x = miu +- b * ln(2*b*p), where p in output of laplace pdf"""
    rng = rng if rng else np.random.default_rng()
    u, r = np.random.uniform(0, 1, (2,) + size)
    x = miu + np.sign(r - 0.5) * b * np.log(u)  # beta range from [ninf, pinf]
    return x


def laplace_quantile(quantile, miu, b):
    if quantile >= 0.5:
        x = miu - b * np.log(2 - 2 * quantile)
    else:
        x = miu + b * np.log(2 * quantile)
    return x
