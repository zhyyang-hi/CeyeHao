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


class ObstacleParameterGenerator:
    def __init__(
        self,
        cw=0.0003,
        num_seg=5,
        x0_range=0.000175,
        mean_sl=0.00015,
        var_sl=0.4,
        name: str = "zigzag",
    ) -> None:
        self.name = name
        self.param = None
        self.meta_param = {
            "cw": cw,
            "num_seg": num_seg,
            "x0_range": x0_range,
            "mean_sl": mean_sl,
            "var_sl": var_sl,
        }

    def y_cal_batch(self, param_list, cw):
        """receive a param_list for obstacles, convert the columns of the y weights into coordinates."""
        ratio_cols = param_list[:, 5::3]
        n_seg = ratio_cols.size(1)
        ratio_cols = ratio_cols * (10 - n_seg) + 4
        sw = ratio_cols / ratio_cols.sum(1, True) * cw

        y_coords = torch.zeros_like(sw)
        for i in range(sw.size(1)):
            y_coords[:, i] = torch.sum(sw[:, 0 : i + 1], 1)

        param_list[:, 5::3] = y_coords
        param_list[:, 2] = 0

        return param_list

    def gen_param(self, num_obs, quasi_rand=True):
        """generate a tensor of parameter list for  obstacle simulation.
        for each obstacle the parameters are: [xn0, xn1, yn for n in linspace(0,5)]
        param meaning:
            xn:     axial position of stramfront obstacle corner, ranging (-1,1)*x0_range
            sln:    obstacle length, ranging (0.5,1.5)*mid_sl
            yn:     y position of each row of corner.

        yn starts sampled as ratio

        """
        num_seg = self.meta_param["num_seg"]
        x0_range = self.meta_param["x0_range"]
        mean_sl = self.meta_param["mean_sl"]
        var_sl = self.meta_param["var_sl"]
        cw = self.meta_param["cw"]

        num_param = (num_seg + 1) * 3
        if quasi_rand:
            soboleng = torch.quasirandom.SobolEngine(dimension=num_param)
            param_list = soboleng.draw(num_obs)
        else:
            param_list = torch.rand((num_obs, num_param))

        # convert y weights into y coordinates
        param_list = self.y_cal_batch(param_list, cw)

        # convert x ratio into coords
        param_list[:, 0::3] = param_list[:, 0::3] * x0_range * 2 - x0_range

        # convert obstacle length ratio into x coordinates of downstream corners
        param_list[:, 1::3] = (
            param_list[:, 1::3] * var_sl * 2 - var_sl + 1
        ) * mean_sl + param_list[:, 0::3]

        # add index to each obstacle
        param_index = torch.arange(0, num_obs).reshape((num_obs, 1))
        param_list = torch.cat((param_index, param_list), 1)
        self.param = param_list
        return param_list

    def draw_obs(self, index):
        """draw obstacles from the paramlist (not coordinates. dont mess up with the coordinates.) in the sample"""
        x0_range = self.meta_param["x0_range"]
        mean_sl = self.meta_param["mean_sl"]
        var_sl = self.meta_param["var_sl"]
        cw = self.meta_param["cw"]

        x_margin = (
            x0_range  # margin prepared for alternating the obstacle axial position
        )
        xlim0 = -x0_range - x_margin
        xlim1 = x0_range + (1 + var_sl) * mean_sl + x_margin
        box_aspect = cw / (xlim1 - xlim0)

        fig = plt.figure(index, figsize=(5, 5 * box_aspect), dpi=600)
        ax = fig.add_axes(
            [0, 0, 1, 1], xlim=[xlim0, xlim1], ylim=[0, cw], aspect="equal"
        )

        p = self.param[index][1:]

        x = torch.cat((p[0::3], torch.flip(p[1::3], dims=(0,))))
        y = torch.cat((p[2::3], torch.flip(p[2::3], dims=(0,))))
        xy = torch.stack((x, y), dim=1)

        ax.add_patch(mpatches.Polygon(xy, closed=True))
        plt.axis("off")
        return fig

    def export_obs_images(self, save_dir, index_list):
        for i in tqdm(list(index_list)):
            fname = f"{save_dir}/{self.name}-{i}.png"
            fig = self.draw_obs(i)
            plt.figure(i)

            plt.savefig(fname)
            plt.close(i)

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


def draw_obs_grid(param_list, index_range, cw):

    if isinstance(param_list, torch.Tensor) is False:
        param_list = torch.from_numpy(param_list)

    p = param_list[:, 1:]
    p = p.reshape((p.size(0), -1, 3))

    n_obs = index_range[1] - index_range[0]

    plt.figure(figsize=[15, 12])
    for i in range(n_obs):
        current_obs = p[index_range[0] + i]
        plt.subplot(
            math.ceil(n_obs / 3), 3, i + 1, xlim=[-1.5 * cw, 1.5 * cw], ylim=[0, cw]
        )
        plt.plot(
            current_obs[:, 0],
            current_obs[:, 2],
            "b",
            current_obs[:, 1],
            current_obs[:, 2],
            "b",
        )
        plt.plot(
            current_obs[0, 0:2].tolist(),
            [current_obs[0, 2], current_obs[0, 2]],
            "b",
            current_obs[-1, 0:2].tolist(),
            [current_obs[-1, 2], current_obs[-1, 2]],
            "b",
        )
        plt.axis("off")
        plt.title(f"{index_range[0]+i}")
    plt.show()


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
        cw,
        res,
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

    def preprocess_single(self, raw_fname, output_dir=None):
        output_dir = output_dir or self.output_dir
        sl_df = self.load_stl(raw_fname)
        sl_mat = self.sl2mat(sl_df, self.cw, self.res)
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
    def sl2mat(pd_sl_list, cw: list, res: list, nan_imputation=True):
        # assert y start is within the channel width
        assert (
            pd_sl_list["y end"].max() <= cw[0] and pd_sl_list["z end"].max() <= cw[1]
        ), "start coord out of channel dimension."

        # remove incomplete streamlines: start/end distance below regular.
        ave_chn_len = np.round(
            pd_sl_list["x end"].max() - pd_sl_list["% x start"].min(), 6
        )
        raw_lite = pd_sl_list[
            (pd_sl_list["x end"] - pd_sl_list["% x start"] >= (ave_chn_len - 5e-6))
        ].iloc[:, [1, 2, 4, 5]]

        # list of boundary of each pixel
        bound_0 = np.linspace(0, cw[0], res[0] + 1)
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
        sl_list[:, 0] = sl_list[:, 0] / (cw[0] / res[0])
        sl_list[:, 1] = sl_list[:, 1] / (cw[1] / res[1])

        sl_mat = np.zeros((res[0], res[1], 2))
        sl_mat[:, :] = None
        for sl in sl_list:
            sl_mat[int(sl[2]), int(sl[3])] = [sl[0], sl[1]]

        # from inlet coord to displacement
        end_coord = np.indices([res[0], res[1]]).transpose([1, 2, 0]) + 0.5
        displ_mat = sl_mat - end_coord

        # remove NaN
        if nan_imputation == True:
            displ_mat = NanImputation(displ_mat)
        return displ_mat


def NanImputation(tt):
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
