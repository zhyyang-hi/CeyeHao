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
    fp_res: list = [200, 200],
    param_scale=True,
    dtype=np.uint8,
    mr=0.,
):
    """generate the input profile matrix and plot from the stream band start/end positions.
    Args:
        param (np.ndarray): shape ((b),2, c) where b is the batchsize of pin and c is the number of stripes in each pin.
            each strips is defiend by the L/R boundary position and exclusively takes a image channel.
        fp_res (list): resolution of pin.
        ax (mpl.axes.Axes): axes to plot on.
        param_scale (bool, optional): if set to False, the param should be given in 0-pf_res. otherwise param is in [0,100).
        dtype (np.dtype, optional): the dtype of the returned tensor. uint8 or bool. Defaults to np.uint8.
    Returns:
        np.ndarray: a batch of pins of shape (b, H, W, c)
    """
    if type(param) != np.ndarray:
        param = np.array(param)

    if param.ndim == 2:
        assert (
            param.shape[0] == 2
        ), f"params of a single input flow profile should be gien in shape (2, c), but got {param.shape}"
        param = param[np.newaxis, :]

    elif param.ndim == 3:
        assert (
            param.shape[1] == 2
        ), f"Pin param should be of shape (b,2,c), but got{param.shape}"
        pass
    else:
        raise ValueError(
            f"incorrect pin param shape, should be ((b),2, c), but got {param.shape}"
        )

    num_fps = param.shape[0]
    num_stripes = param.shape[2]

    if param_scale:
        band_value_scale = fp_res[0] / 100
        param = (param * band_value_scale).astype(int)
    # axes aligned with y/z axes of microchannel space, not the axes of image
    pins = np.zeros((num_fps, fp_res[0], fp_res[1], num_stripes), dtype=dtype)
    for i in range(num_fps):
        for j in range(num_stripes):
            pins[i, param[i, 0, j] : param[i, 1, j], :, j] = 255

    if mr != 0:
        new_w = int(fp_res[0]*(1+abs(mr)))
        dw = int(fp_res[0]*abs(mr))
        m_pin = np.zeros((num_fps, new_w, fp_res[1], num_stripes), dtype=dtype)
        if mr<0:
            m_pin[:, :new_w-dw, fp_res[1]//2:, :] = pins[:,:, fp_res[1]//2:, :]
            m_pin[:, dw:, :fp_res[1]//2, :] = pins[:,:, :fp_res[1]//2, :]
        elif mr>0:
            m_pin[:, :new_w-dw, :fp_res[1]//2, :] = pins[:,:, :fp_res[1]//2, :]
            m_pin[:, dw:, fp_res[1]//2:, :] = pins[:,:, fp_res[1]//2:, :]
        pins = m_pin


    return pins


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
    pins pf shape (H,W,...)
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
    """computes the flow rate ratio or flow width ratio of each stripe in the inflow profile.
    consider a flow velocity parabollic curve with roots 0 and 1 (i.e. chn width =1), with total flow rate =1
    """

    def __init__(self, verbose=True):
        self.verbose = verbose

    def area_func(self, x):
        return 3 * x**2 - 2 * x**3

    def coeff(self, x):
        # coefficients of the area functin to solve
        return np.stack([np.full_like(x, -2), np.full_like(x, 3), np.zeros_like(x), -x])

    def calc_fr_ratio(self, wr1, wr2, wr3):
        """wr1 ~ wr3 given in the w:0->1 direction"""
        # compute the area betwwen wi
        wr1, wr2, wr3 = np.array((wr1, wr2, wr3))
        assert (wr1 + wr2 + wr3 == 1).all(), "the width ratio should sum to 1"

        frr1 = self.area_func(wr1)
        frr2 = self.area_func(wr1 + wr2) - frr1
        frr3 = 1 - frr1 - frr2
        frr1, frr2, frr3 = np.array((frr1, frr2, frr3)).round(3)
        if self.verbose:
            print(f"flow rate ratio: {frr1, frr2, frr3}")
        return frr1, frr2, frr3

    def calc_wr_ratio(self, frr1, frr2, frr3):
        frr1, frr2, frr3 = np.array((frr1, frr2, frr3))
        assert ((frr1 + frr2 + frr3).round(6) == 1.).all(), "flow rate ratio should sum to 1"
        # assert rounded to aoivd float computation error
        wr1 = np.apply_along_axis(np.roots, 0, self.coeff(frr1))[1]
        wr2 = np.apply_along_axis(np.roots, 0, self.coeff(frr1 + frr2))[1] - wr1
        wr3 = 1 - wr1 - wr2
        wr1, wr2, wr3 = np.array([wr1, wr2, wr3]).round(3)
        if self.verbose:
            print(f"stripe width ratio: {wr1, wr2, wr3}")
        return wr1, wr2, wr3

    def find_r_bound(self, l_bound, frr):
        frr_l = self.area_func(l_bound)
        r_bound = np.apply_along_axis(np.roots, 0, self.coeff(frr_l + frr))[1]
        return r_bound

    @staticmethod
    def estimate_fr_ratio(profile: np.ndarray, verbose=False):
        """compute the flow rate ratio of a arbitrary profile to the total flow rate.
        if input profile is multi-channel, profile is binarized first.
        ATTENTION: the 2D velocity profile is approximated by v = Cx(1-x)y(1-y). Error exists.
        Args:
            profile (np.ndarray): the flow profile, shape (H,W (,C)).
        """
        #NOTE treat 3 channel image as single channel.
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
        if verbose:
            print(f"flow rate ratio of target flow profile is {profile_fr / total_fr}")
        return profile_fr / total_fr


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

def img2fp_coords(img:np.ndarray):
    """input/output shape (h,w,c). change the img coordinate system to 
        microchannel x-section coordiante system. """
    return np.swapaxes(img,0,1)[:, ::-1,]
    # return img.transpose([1, 0, 2])[:, ::-1, :]


