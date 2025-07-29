import os, sys

parent_path = os.path.abspath(os.path.join(__file__, *([".."] * 2)))
sys.path.insert(0, parent_path)
import torch
import numpy as np
import matplotlib.pyplot as plt

from ceyehao.models.archs import *
from ceyehao.tools.trainer import Trainer
from ceyehao.data.transform import TransformObsImg
from ceyehao.utils.data_process import tt_convert, tt_postprocess, obs_params2imgs
from ceyehao.utils.visualization import (
    plot_tt_vf,
    create_obstacle_figure,
)
from ceyehao.utils.io import load_cfg_yml


class TTPredictor(Trainer):
    """take raw data, apply transform, return the ready-to-use transformation tensor(s)."""

    def __init__(self, cfg=None):
        #TODO retrieve the default config and model weights from online repository
        if cfg is None:  # create a default config
            cfg = load_cfg_yml("../log/CEyeNet/infer_cfg.yml")
        assert cfg.mode == "infer"
        super(TTPredictor, self).__init__(cfg)
        self.transform = TransformObsImg(self.cfg)
        self.obs_fig, self.obs_ax = create_obstacle_figure()

    def predict_from_obs_param(self, params, pos, postprocess=True):
        """

        Args:
            params (ndarray): shape=[(B), 18]
            pos (ndarray): shape = [(B), 1]. dtype=bool
            transform (bool, optional): _description_. Defaults to True.
            postprocess (bool, optional): process tt to functionable. Defaults to True.

        Returns:
            tts (ndarray): shape [(B), H, W, 2]
        """
        obs_imgs = obs_params2imgs(params, pos, self.obs_ax)
        tts = self.predict_from_obs_img(obs_imgs, postprocess=postprocess)
        tts = tt_convert(tts, vert_sym=pos)
        return tts

    def predict_from_obs_img(self, img, transform=True, postprocess=True):
        """predict the transformation tensor (numpy arrary, ready-to-use) of the input image.
        Args:
            img: np.ndarray, shape=[(B), H, W, 3]
            transform: bool, whether to apply transform to the input image.
            postprocess: bool, whether to apply postprocess to the output transformation tensor.
        """

        with torch.no_grad():

            if len(img.shape) == 3:
                img = img[np.newaxis, ...]
            if transform == True:
                transformed_img = []
                for i in img:
                    transformed_img.append(self.transform(i))
                img = torch.stack(transformed_img)
            img = img.to(self.cfg.device)
            with torch.autocast(
                device_type=self.cfg.device, dtype=torch.float16, enabled=self.cfg.amp
            ):
                output = self.model(img)
            if self.cfg.model == "gvtn":
                output = output[1]
            # squeeze if only output transforamtion tensor
            output = output.squeeze().cpu()
            if postprocess:
                output = tt_postprocess(output)
        return output


class Plotter:
    """Take x, y, yhat, plot in the same figure."""

    def __init__(
        self,
        save_dir="./tmp",
        plot_res=[40, 40],
        figsize=[5, 2],
        dpi=300,
    ):
        self.plot_res = plot_res
        self.save_dir = save_dir
        self.figsize = figsize
        self.dpi = dpi
        pass

    def __call__(self, x, y, y_hat, res_idx, namestring=""):
        fig, axes = self.plot_preds(x, y, y_hat)
        plt.savefig(self.save_dir + "/" + str(res_idx).zfill(4) + namestring + ".jpg")
        plt.close("all")

    def select_n_rand(self, x, y, n=3):
        perm = torch.randperm(len(x))
        return x[perm][:n], y[perm][:n]

    def plot_preds(self, x, y, y_hat):

        fig, axes = plt.subplots(1, 3, figsize=self.figsize, dpi=self.dpi)
        axes[0].set_title("x")
        axes[1].set_title("y")
        axes[2].set_title("y_hat")
        axes[0].imshow(x)
        axes[0].axis("off")
        plot_tt_vf(y.transpose(1, 2, 0), axes[1], plot_res=self.plot_res)
        plot_tt_vf(y_hat.transpose(1, 2, 0), axes[2], plot_res=self.plot_res)
        return fig, axes
