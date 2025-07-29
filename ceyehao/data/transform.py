import numpy as np
import torchvision.transforms as tvtf
import torch
from torchvision.transforms import InterpolationMode

from ceyehao.utils.data_process import tt_convert

class TransformObsImg:
    """transform obstacle image to Greyscale and resize to the neural input size tar_res, and optional flip"""

    def __init__(self, cfg) -> None:
        self.tar_res = cfg.simg_res
        self.axial_flip = cfg.data_cfg.x_axial_flip if hasattr(cfg.data_cfg, "x_axial_flip") else None
        self.randinv = cfg.data_cfg.x_randinv if hasattr(cfg.data_cfg, "x_randinv") else None
        self.randaxtl = cfg.data_cfg.x_rand_axial_translate if hasattr(cfg.data_cfg, "x_rand_axial_translate") else None
        self.compeye = cfg.data_cfg.compeye if hasattr(cfg.data_cfg, "compeye") else None
        self.translator = tvtf.RandomAffine(0, translate=[0.2, 0], fill=0.9999999404)
        self.inverter = tvtf.RandomInvert()
        self.resizer = tvtf.Resize(
            self.tar_res, InterpolationMode.BILINEAR, antialias=True
        )
        self.grayscaler = tvtf.Grayscale()

    def __call__(self, img, axial_sym_trigger=False):
        """inpput image should be in the follwoing formats:
        numpy array ranging 0~255 in shape (H,W,C), color RGB
        numpy array ranging 0~255 in (H,W), color grayscale
        torch tensor ranging 0~1 in (C, H, W), color RGB or grayscale
        """

        if type(img) == np.ndarray:
            img = torch.from_numpy(img.copy()) / 255
            if len(img.shape) == 3:
                img = img.permute(2, 0, 1).contiguous()
            if len(img.shape) == 2:
                img = img[np.newaxis, ...]

        img = self.resizer(img)
        img = self.grayscaler(img)
        img = torch.round(img, decimals=6)
        img = tvtf.functional.autocontrast(img)

        if self.randaxtl == True:
            img = self.translator(img)  # filll with img max to avoid black edge
        if self.axial_flip == True:
            img = tvtf.functional.hflip(img)
        if self.randinv == True:
            img = self.inverter(img)
        if axial_sym_trigger == True:
            img = tvtf.functional.vflip(img)

        if self.compeye == True:
            imgh = torch.cat((img, img, img), dim=-2)
            img = torch.cat((imgh, imgh, imgh), dim=-1)
            img = self.resizer(img)

        return img


"""
    def forward(self, img):

        for t in self.transforms:
            img = t(img)
        return img
"""


class TransformTT:

    def __init__(self, cfg) -> None:
        self.tar_res = cfg.profile_size

    def __call__(self, tt, sym_trigger=False):

        if sym_trigger == True:
            tt = tt_convert(tt, horiz_sym=True)

        return tt
