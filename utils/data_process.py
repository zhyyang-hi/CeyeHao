import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
import torch
from typing import Union

from utils.visualization import plot_obstacle, create_obstacle_figure, fig2np


def gen_pin_tensor(
    param: Union[np.ndarray, list],
    pf_res: list = [200, 200],
    param_scale=True,
    dtype=np.uint8 
):
    """generate the input profile matrix and plot from the stream band start/end positions.
    Args:
        param (np.ndarray): (band1_left_bound, band2_left_bound, ...)
        res (list): resolution of the input file
        ax (mpl.axes.Axes): axes to plot on.
        param_scale (bool, optional): if set to False, the param should be given in 0-pf_res. otherwise param is in 0-99。
        dtype (np.dtype, optional): the dtype of the returned tensor. uint8 or bool. Defaults to np.uint8.
    """
    if type(param) == list:
        param = np.array(param)
    if param_scale:
        band_value_scale = pf_res[0] / 100
        param = (param * band_value_scale).astype(int)
    # axes aligned with y/z axes of microchannel space, not the axes of image
    if len(param) == 2:
        pin_mat = np.zeros((pf_res[0], pf_res[1], 1), dtype=dtype)
        pin_mat[param[0] : param[1]] = 255  # G
    else:
        pin_mat = np.zeros((pf_res[0], pf_res[1], 3), dtype=dtype)
        pin_mat[param[0] : (param[1]), :, 0] = 255  # R
        pin_mat[param[2] : (param[3]), :, 1] = 255  # G
        pin_mat[param[4] : (param[5]), :, 2] = 255  # B
    

    return pin_mat


def tt_postprocess(tt: np.ndarray | torch.Tensor):
    """process raw or predicted transformation tensor(s) to standard form:
    type to ndarray
    round the tensor to integer
    shape to (y,z,2).
    """
    if type(tt) == torch.Tensor:
        tt = tt.detach().cpu().numpy()

    if tt.dtype is not np.dtype("int16"):
        tt = tt.round().astype(np.int16)

    if tt.shape[-1] == 2:
        pass
    elif tt.shape[0] == 2:
        tt = np.rollaxis(tt, 0, tt.ndim)
    elif tt.shape[1] == 2:
        tt = np.rollaxis(tt, 1, tt.ndim)
    else:
        raise ValueError(f"wrong tt shape: {tt.shape}")
    return tt


def p_transform(pin: np.ndarray, tts: np.ndarray | torch.Tensor, full_p_records=False):
    """Compute the output profile given a transformation matrix and input profile
    tts of shape of (H,W,2) or (n,H,W,2)
    """
    assert tts.ndim in [3, 4], "number of dimensions of tt(s) should be 3 or 4"
    assert tts.shape[-1] == 2, "tt shape should be (H,W,2)"
    assert pin.shape[0:2] == tts.shape[-3:-1], "input and tt should have same H and W"
    res = tts.shape[-3:-1]
    if tts.ndim == 3:
        tts = tts[np.newaxis, :]

    tts = tt_postprocess(tts)
    p = np.zeros_like(pin)
    pout = np.stack([pin] + [p] * tts.shape[0], axis=0)
    for i, tt in enumerate(tts):
        tt = tt + np.mgrid[0 : res[0], 0 : res[1]].transpose([1, 2, 0])
        # clip index-out-of-range element values
        tt[:, :, 0] = np.clip(tt[:, :, 0], 0, res[0] - 1)
        tt[:, :, 1] = np.clip(tt[:, :, 1], 0, res[1] - 1)
        pout[i + 1] = pout[i, tt[:, :, 0].astype(int), tt[:, :, 1].astype(int)]
    if full_p_records:
        return pout
    else:
        return pout[-1]


def tt_convert(
    tts: np.ndarray,
    horiz_sym: Union[list, bool, np.ndarray] = None,
    vert_sym: Union[list, bool, np.ndarray] = None,
) -> np.ndarray:
    """transform the tt according to the transformation of the obstacle
    tts of shape ((B,) H,W,2)."""
    original_ndim = tts.ndim

    if original_ndim == 3:
        new_tts = tts[np.newaxis, :]
    elif original_ndim == 4:
        new_tts = tts.copy()

    if vert_sym is not None:
        vert_sym = np.array(vert_sym, dtype=bool)
        if vert_sym.ndim == 0:
            vert_sym = np.array([vert_sym])
        new_tts[vert_sym] = np.flip(new_tts[vert_sym], axis=-2)
        new_tts[vert_sym, :, :, 1] = -new_tts[vert_sym, :, :, 1]
    if horiz_sym is not None:
        horiz_sym = np.array(horiz_sym, dtype=bool)
        if horiz_sym.ndim == 0:
            horiz_sym = np.array([horiz_sym])
        new_tts[horiz_sym] = np.flip(new_tts[horiz_sym], axis=-3)
        new_tts[horiz_sym, :, :, 0] = -new_tts[horiz_sym, :, :, 0]
    if original_ndim == 3:
        new_tts = new_tts[0]
    return new_tts


def tt_reverse(tt: np.ndarray) -> np.ndarray:
    """revsersed flow direction"""
    rev = np.empty(tt.shape)
    res = tt.shape[0:2]
    coords = np.mgrid[0 : res[0], 0 : res[1]].transpose([1, 2, 0])
    tt_abs = tt + coords
    tt_abs[:, :, 0] = np.clip(tt_abs[:, :, 0], 0, res[0])
    tt_abs[:, :, 1] = np.clip(tt_abs[:, :, 1], 0, res[1])
    for i in range(res[0]):
        for j in range(res[1]):
            rev[tt_abs[i, j, 0], tt_abs[i, j, 1]] = -tt[i, j]
    return rev


def obs_params2imgs(obs_coords, obs_pos, ax=None):
    """_summary_

    Args:
        obs_coords (ndarray): of shape ((B,) 18)
        obs_pos (bool, optional): of shape ((B,) 1)
        ax (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if ax is None:
        flag_close_fig = True
        fig, ax = create_obstacle_figure()
    else:
        fig = ax.get_figure()
        flag_close_fig = False

    if obs_coords.ndim == 1:
        obs_coords = obs_coords[np.newaxis, :]
        obs_pos = obs_pos[np.newaxis, :]

    obs_imgs = []
    for param, pos in zip(obs_coords, obs_pos):
        plot_obstacle(param, pos, ax)
        img = fig2np(fig)
        obs_imgs.append(img)

    if flag_close_fig:
        plt.close(ax.get_figure())
    return np.stack(obs_imgs, axis=0)


def tt_synth(tt_list: list, interm=False) -> np.ndarray:
    """compute the resultant transformation matrix(tt) from a list of tms.
    Args:
        tt_list (list): list of transformation matrices, shape (H,W,2)
        interm (bool, optional): if True, return the intermediate tts. Defaults to False.
    """
    res = tt_list[0].shape[0:2]
    y_list = np.arange(res[0])
    z_list = np.arange(res[1])
    z, y = np.meshgrid(y_list, z_list)
    grids = np.mgrid[0 : res[0], 0 : res[1]].transpose([1, 2, 0])
    coord_tans = grids.copy()

    net_tts = np.array(tt_list)

    for i, tt in enumerate(tt_list):
        coord_tans = p_transform(coord_tans, tt)
        net_tts[i] = coord_tans[:]
    net_tts = net_tts - grids
    net_tts = net_tts.astype(np.int16)
    if interm:
        return net_tts
    else:
        return net_tts[-1]


def obs_param_convert(stp: np.ndarray, cw=3e-4):
    """compute the coordinats of the new obstacle mirrored in channel lateral direction"""
    ori_shape = stp.shape
    new = stp.reshape(6, 3)
    new[:, 2] = cw - new[:, 2]
    new[:] = new[::-1]
    new = new.reshape(ori_shape)
    return new


class InflowCalculator:
    """calculate the inflow rate of each inlet branch to create a target inlet flow profile"""

    def __init__(self, chn_width, total_flow_rate, verbose=True):
        self.chn_width = chn_width
        self.total_flow_rate = total_flow_rate  # the physical flow rates.
        self.total_fr = self.area_under_curve(
            0, chn_width
        )  # i.e. total flow rate, for computation
        self.verbose = verbose
        self.flow_rates = np.array(
            [1 / 3 * total_flow_rate, 1 / 3 * total_flow_rate, 1 / 3 * total_flow_rate]
        )
        self.set_flow_rates(self.flow_rates[0], self.flow_rates[1], self.flow_rates[2])

    def print_info(self):
        print(f"\nchannel width in pixel: {self.chn_width}")
        print(f"total flow rate: {self.total_flow_rate}")
        print(
            f"flow rates: {self.flow_rates[0]:.0f}, {self.flow_rates[1]:.0f}, {self.flow_rates[2]:.0f}"
        )
        print(
            f"flow rate ratio: {self.flow_rates_ratio[0]:.3f}, {self.flow_rates_ratio[1]:.3f}, {self.flow_rates_ratio[2]:.3f}"
        )
        print(
            f"stream width:{self.widths[0]:.3f}, {self.widths[1]:.3f}, {self.widths[2]:.3f}"
        )
        print(
            f"stream interfaces: {self.widths[0]:.3f}, {self.widths[0]+self.widths[1]:.3f}"
        )

    def set_flow_rates(self, fr1, fr2, fr3):
        """given the inflow rate of each stream, find the width of each stream

        Args:
            fr1 (float): flow rate of stream 1
            fr2 (float): flow rate of stream 2
            fr3 (float): flow rate of stream 3
            chn_width (int, optional): width of channel. Defaults to 1.
        """
        self.total_flow_rate = fr1 + fr2 + fr3
        p1 = fr1 / (fr1 + fr2 + fr3)
        p2 = fr2 / (fr1 + fr2 + fr3)
        p3 = fr3 / (fr1 + fr2 + fr3)
        w1 = self.find_target_x(self.chn_width, p1)
        w3 = self.find_target_x(self.chn_width, p3)
        w2 = self.chn_width - w1 - w3
        self.flow_rates = np.array([fr1, fr2, fr3])
        self.flow_rates_ratio = np.array([p1, p2, p3])
        self.widths = np.array([w1, w2, w3])
        if self.verbose:
            self.print_info()
        return w1, w2, w3

    def set_flow_widths(self, w1, w2, w3):
        """given the target width of each stream, find the ratio of flow rate of each stream"""
        self.widths = np.array([w1, w2, w3])
        self.chn_width = w1 + w2 + w3
        area = self.area_under_curve(0, self.chn_width, self.chn_width)
        frr1 = self.area_under_curve(0, w1, self.chn_width) / area
        frr3 = self.area_under_curve(0, w3, self.chn_width) / area
        frr2 = 1 - frr1 - frr3
        self.flow_rates_ratio = np.array([frr1, frr2, frr3])
        self.flow_rates = self.flow_rates_ratio * self.total_flow_rate
        if self.verbose:
            self.print_info()

    def set_chn_width(self, chn_width):
        w1, w2, w3 = self.widths[:] / self.chn_width * chn_width
        self.chn_width = chn_width
        self.set_flow_widths(w1, w2, w3)

    def set_total_flow_rate(self, total_flow_rate):
        self.total_flow_rate = total_flow_rate
        self.flow_rates = self.flow_rates_ratio * total_flow_rate
        if self.verbose:
            self.print_info()

    def parabolic_curve(self, x):
        return -x * (x - self.chn_width)

    def area_under_curve(self, a, b):
        integral, _ = quad(self.parabolic_curve, a, b)
        return integral

    def find_target_x(self, root1, p):
        if not (0 <= p <= 1):
            raise ValueError("Invalid input. Please make sure 0 <= p <= 1 ")

        area = self.area_under_curve(0, self.chn_width)

        def objective_function(t):
            return np.abs(self.area_under_curve(0, t) - p * area)

        res = minimize_scalar(
            objective_function, bounds=(0, self.chn_width), method="bounded"
        )
        return res.x

    @staticmethod
    def compute_pf_vol_ratio(profile: np.ndarray):
        """compute the flow rate ratio of a arbitrary profile to the total flow rate.
        if input profile is multi-channel, profile is binarized first.
        Args:
            profile (np.ndarray): the flow profile, shape (H,W (,C)).
        """
        if profile.ndim == 3:
            profile = profile.sum(axis=-1)
            profile = profile > 0

        res = profile.shape[0]
        # assume channel width is 100, compute the center value of each pixel
        coord_list = np.linspace(100 / res * 0.5, 100 - 100 / res * 0.5, res)

        def parabolic_curve(x):
            return -x * (x - 100)

        value_list = parabolic_curve(coord_list)
        weight_matrix = value_list[:, np.newaxis] * value_list[np.newaxis, :]
        total_fr = weight_matrix.sum()
        profile_fr = (profile * weight_matrix).sum()
        return profile_fr / total_fr

    def find_pin_bound(self, flow_rate_ratio, pin_l=None, pin_r=None):
        if pin_r is None and pin_l is None:
            pin_l = 0
            print("pin_start is not specified, set to 0, try find pin_end")

        if pin_r is None:
            theorectical_max_fr = self.area_under_curve(pin_l, self.chn_width)
            if theorectical_max_fr < flow_rate_ratio * self.total_fr:
                bound = self.chn_width
            else:
                obj = lambda x: abs(
                    self.area_under_curve(pin_l, x) - flow_rate_ratio * self.total_fr
                )
                bound = minimize_scalar(
                    obj, bounds=(pin_l, self.chn_width), method="bounded"
                ).x

        else:
            theorectical_max_fr = self.area_under_curve(0, pin_r)
            if theorectical_max_fr < flow_rate_ratio * self.total_fr:
                pin_l = 0
            else:
                obj = lambda x: abs(
                    self.area_under_curve(x, pin_r) - flow_rate_ratio * self.total_fr
                )
                bound = minimize_scalar(obj, bounds=(0, pin_r), method="bounded").x
        return bound


def pf2cv2_img(img: np.ndarray, tar_res=[200, 200]):
    """process the input profile image array to standard form.
    Args:
        array (np.ndarray): the input profile image array, shape (H,W) or (H,W,3) or (H, W, 1)
        tar_res (int, optional): the target resolution. Defaults to 200.
    """
    if img.dtype in [np.float32, np.float64]:
        img = (img * 255).astype(np.uint8)
    if len(img.shape) == 2:
        empty_chn = np.zeros_like(img)
        img = cv2.merge([empty_chn, img, empty_chn])  # BGR
    elif len(img.shape) == 3:
        if img.shape[2] == 1:
            empty_chn = np.zeros_like(img)
            img = np.concatenate([empty_chn, img, empty_chn], axis=-1)

    img = cv2.resize(img, tar_res, interpolation=cv2.INTER_LINEAR)
    return img

