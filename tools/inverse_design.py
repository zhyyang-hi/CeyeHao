import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
import pandas as pd
import datetime
from easydict import EasyDict

from tools.infer import TTPredictor
from tools.acc_metrics import PerceptualAccuracy, PixelwiseAccuracy
from utils.utils import mkdirs
from utils.data_generate import ObstacleParameterGenerator
from utils.data_process import (
    p_transform,
    obs_params2imgs,
    gen_pin_tensor,
    InflowCalculator,
    pf2cv2_img,
)
from utils.visualization import (
    create_obstacle_figure,
    create_profile_figure,
    plot_pf_tensor,
)

CHN_PER_BATCH = 4
PERCEP_THRES = 0.7
EDGE_THRES = 0.05
FILL_THRES = 0.4


class MirochannelDesignSearcher:
    def __init__(self, infer_cfg) -> None:
        self.res = infer_cfg.profile_size[0]
        self.obs_param_generator = ObstacleParameterGenerator()
        self.obs_fig, self.obs_ax = create_obstacle_figure()
        self.predictor = TTPredictor(infer_cfg)
        self.edge_metric = PixelwiseAccuracy(
            EasyDict({"pix_acc_op_flags": ["edge", "IoU"]})
        )
        self.fill_metric = PixelwiseAccuracy(EasyDict({"pix_acc_op_flags": ["IoU"]}))
        self.percep_metric = PerceptualAccuracy()
        self.infl_calc = InflowCalculator(self.res, 1234, verbose=False)

    def sample_obs(self, num_obs: int = 8):
        obs_params = self.obs_param_generator.gen_param(num_obs, quasi_rand=False)[
            :, 1:
        ]
        obs_pos = np.random.randint(0, 2, num_obs)
        return obs_params, obs_pos

    def sample_pin(self, num_pin: int = 16):
        pin_l = np.random.randint(0, self.max_pin_l, num_pin)
        pin_fr_ratio = np.random.uniform(self.min_fr_ratio, self.max_fr_raio, num_pin)
        pin_r = [
            self.infl_calc.find_pin_bound(pfrr, pl)
            for pfrr, pl in zip(pin_fr_ratio, pin_l)
        ]
        # compute the flow volume and adjust the width
        pin = np.stack([pin_l, pin_r], axis=1).round().astype(int)
        return pin

    def set_task(self, tgt: np.ndarray, max_chn=None, max_obs=8, result_dir="./"):
        tgt = pf2cv2_img(tgt)  # shape H,W,3
        tgt = self.process_target_pf(tgt)

        self.tgt = np.repeat(tgt[np.newaxis], max_obs, axis=0)
        self.max_obs = max_obs
        if max_chn is None:
            max_chn = 99999
        self.total_batches = max_chn // CHN_PER_BATCH * self.max_obs + 1
        self.result_dir = mkdirs(result_dir)
        # 1. compute the flow volume percentage of the profile
        flow_rate_ratio = self.infl_calc.compute_pf_vol_ratio(tgt)
        self.max_fr_raio, self.min_fr_ratio = np.array([1.3, 0.7]) * flow_rate_ratio
        # 2. compute the maximum pin left bound based on the flow volume
        self.max_pin_l = self.infl_calc.find_pin_bound(
            self.min_fr_ratio, pin_r=self.res
        )
        # 3. compute the pin_end position-range based on the sampled pin_start position

    def run(self):
        found = 0
        acc_thresholds = np.array([PERCEP_THRES, EDGE_THRES, FILL_THRES])
        search_progress = tqdm(
            range(self.total_batches), desc=f"found:{found}", leave=False
        )
        for batch_id in search_progress:  # to be multi-threaded
            pin_params = self.sample_pin(CHN_PER_BATCH)
            obs_coords, obs_pos = self.sample_obs(
                CHN_PER_BATCH * self.max_obs
            )  # shape (num_chn*num_obs, n)

            tts = self.predictor.predict_from_obs_param(obs_coords, obs_pos)
            for chn_id in tqdm(
                range(CHN_PER_BATCH), leave=False, desc=f"evaluating profiles"
            ):
                pin = gen_pin_tensor(pin_params[chn_id], param_scale=False)
                profiles = p_transform(
                    pin,
                    tts[chn_id * self.max_obs : (chn_id + 1) * self.max_obs],
                    full_p_records=True,
                )[
                    1:
                ]  # drop the input profile
                edge_acc = self.edge_metric([profiles, self.tgt])  # shape (num_obs)
                fill_acc = self.fill_metric([profiles, self.tgt])
                percep_acc = self.percep_metric([profiles, self.tgt])
                # save sequence and pin yiedling satisfactory results
                accs = np.stack([percep_acc, edge_acc, fill_acc], axis=1)

                good_obs_ids = np.argwhere((accs > acc_thresholds).prod(axis=-1))[
                    :, 0
                ]  # keep only the obs position
                good_obs_ids = np.unique(good_obs_ids)

                if len(good_obs_ids) > 0:
                    found += 1
                    search_progress.set_description(f"found:{found}")
                    # name the flow system with datetime
                    flowsys_name = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
                    last_obs_id = good_obs_ids[-1]
                    acc_thresholds = (
                        accs[good_obs_ids].reshape(-1, 3).mean(axis=0)
                    )  # elevate the threshold
                    search_progress.set_description(
                        f"found:{found} | {acc_thresholds.round(3)}"
                    )
                    chn_pin = pin_params[chn_id]
                    chn_obs_coords = obs_coords[
                        chn_id * self.max_obs : (chn_id + 1) * self.max_obs
                    ]
                    chn_obs_pos = obs_pos[
                        chn_id * self.max_obs : (chn_id + 1) * self.max_obs
                    ]
                    chn_accs = accs[: last_obs_id + 1]
                    obs_imgs = obs_params2imgs(chn_obs_coords, chn_obs_pos)
                    # 1. record: obs_param, obs_pos to {datetime}_{last_obs_id}_obs.csv file (for storage and cad drawing)
                    df_chn_obs_coords = pd.DataFrame(
                        chn_obs_coords,
                        columns=[
                            it
                            for ls in [(f"x{i}0", f"x{i}1", f"y{i}") for i in range(6)]
                            for it in ls
                        ],
                    )
                    df_chn_obs_pos = pd.DataFrame(
                        chn_obs_pos[..., np.newaxis], columns=[f"pos"]
                    )
                    df_chn_obs = pd.concat(
                        [df_chn_obs_coords.round(7), df_chn_obs_pos], axis=1
                    )
                    df_chn_obs.to_csv(
                        os.path.join(self.result_dir, f"{flowsys_name}_obs.csv"),
                        index=True,
                        index_label="obs_id",
                    )
                    # 2. pin and intermediata profile acc to {flowsys_name}_{obs_id}_pin-acc.csv.
                    df_chn_accs = pd.DataFrame(
                        chn_accs.round(3),
                        columns=["percep", "edge", "fill"],
                        index=[f"obs_{i}" for i in range(last_obs_id + 1)],
                    )
                    df_chn_accs.to_csv(
                        os.path.join(self.result_dir, f"{flowsys_name}_acc.csv"),
                        index=True,
                        index_label="metric",
                    )
                    df_chn_pin = pd.DataFrame(
                        chn_pin[np.newaxis, ...], columns=["pin_start", "pin_end"]
                    )
                    df_chn_pin.to_csv(
                        os.path.join(self.result_dir, f"{flowsys_name}_pin.csv"),
                        index=True,
                    )

                    # 3. obstacles to obstacle_img/{batch_id}_{chn_id}_{obs_id (multiple)}.png
                    obs_img_dir = mkdirs(os.path.join(self.result_dir, "obstacle_img"))
                    for i, img in enumerate(obs_imgs):
                        cv2.imwrite(
                            os.path.join(obs_img_dir, f"{flowsys_name}_obs_{i}.png"),
                            img,
                        )
                    # 4. profile figures to profile_img/{batch_id}_{chn_id}_{obs_id}_{metric}-{metric value}.png
                    pf_dir = mkdirs(os.path.join(self.result_dir, "profile_img"))
                    fig, ax = create_profile_figure()
                    for i in good_obs_ids:
                        max_acc_value = df_chn_accs.loc[f"obs_{i}"].max()
                        max_acc_id = df_chn_accs.loc[f"obs_{i}"].idxmax()
                        pf_fname = (
                            f"{flowsys_name}_pf_{i}_{max_acc_id}_{max_acc_value}.png"
                        )
                        plot_pf_tensor(profiles[i], ax)
                        fig.savefig(
                            os.path.join(
                                pf_dir,
                                pf_fname,
                            )
                        )
                    plt.close(fig)
        print(f"Found {found} satisfactory flow systems.")

    def process_target_pf(self, pf: np.ndarray):
        """
        process the target profile to the binarized single channel format
        Args:
            pf (np.ndarray): the target profile image, shape=(H, W, 3) dtype=uint8

        """
        pf = pf.sum(axis=-1, keepdims=True)  # shape=(H, W, 1)
        pf = pf.clip(0, 255).astype(np.uint8)
        return pf
    
    def __call__(self, target_profile=r"../auto_search/target_profile.png", result_dir=r"../auto_search"):
        self.set_task(target_profile=target_profile, result_dir=result_dir)
        self.run()
        
