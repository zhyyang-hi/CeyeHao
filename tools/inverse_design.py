import numpy as np
import matplotlib.pyplot as plt
import os
import torch.multiprocessing as mp
from easydict import EasyDict
from queue import Empty
import time
from tqdm import tqdm

from tools.infer import TTPredictor
from tools.acc_metrics import PerceptualAccuracy, PixelwiseAccuracy
from utils.utils import mkdirs, get_logger, Timer
from utils.data_generate import HAOParamSampler
from utils.data_process import (
    p_transform,
    gen_pin_tensor,
    InflowCalculator,
    tt_synth,
    img2fp_coords
)
from utils.visualization import (
    create_profile_figure,
    plot_fp_tensor,
)


class InvDesign:
    def __init__(self, fp_res=(200, 200)) -> None:
        self.fp_res = fp_res
        self.hao_coord_sampler = HAOParamSampler()
        self.inp_calc = InflowCalculator(verbose=False)

    def set_task(
        self,
        tgt_fp: np.ndarray,
        num_chn,
        chn_batch_size,
        num_hao,
        pin_per_chn,
        restrict_flow_ratio,
        acc_thresholds,
        log_dir,
    ):
        """
        pf is single channel, uint8"""
        assert (
            tgt_fp.shape[0:2] == self.fp_res
        ), f"the target image size is{tgt_fp.shape[0:2]}, but the expected size is {self.fp_res}"

        self.log_dir = log_dir
        self.num_chn = num_chn
        self.num_hao = num_hao
        self.pin_per_chn = pin_per_chn
        self.chn_bsize = chn_batch_size
        self.num_batch = (
            num_chn // chn_batch_size
            if num_chn % chn_batch_size == 0
            else num_chn // chn_batch_size + 1
        )
        self.acc_thresholds = np.array(acc_thresholds)
        if restrict_flow_ratio:
            target_flow_ratio = InflowCalculator.estimate_fr_ratio(tgt_fp)
            self.fr_range = (target_flow_ratio * np.array([0.7, 1.3])).clip(0.1, 0.8)
        else:
            self.fr_range = (0.1, 0.8)
        tgt_fp = tgt_fp[..., np.newaxis]
        tgt_fp = img2fp_coords(tgt_fp)
        self.target_fp = np.tile(
            tgt_fp,
            [
                pin_per_chn * num_hao,
                *[1] * tgt_fp.ndim,
            ],  # shape (pin_per_chn * num_obs, H, W, C)
        )  # shape (pin_per_chn * num_obs, H, W, (C))
        fig, _ = plot_fp_tensor(tgt_fp)
        fig.savefig(os.path.join(log_dir, 'tgt_img.png'))

    def run_task(self, parallel=1, log_level=20):
        """
        run the task and save the results
        """
        timer = Timer(verbose=False)

        img_dir = mkdirs(os.path.join(self.log_dir, "fp_img"))
        param_dir = mkdirs(os.path.join(self.log_dir, "params"))
        # save the metadata
        metadata = {
            "fp_res": self.fp_res,
            "num_chn": self.num_chn,
            "num_obs": self.num_hao,
            "pin_per_chn": self.pin_per_chn,
            "fr_range": self.fr_range.tolist(),
            "acc_metrics": "0-IoU 1-Edge_IoU",
            "acc_thresholds": self.acc_thresholds.tolist(),
            "parallel worker": parallel,
            "channel per batch": self.chn_bsize,
            "total_batches": self.num_batch,
        }

        for batch_id in range(self.num_batch):
            self.sample_params(batch_id)
            manager = mp.Manager()
            tt_queue = manager.Queue()
            obs_param_queue = manager.Queue()
            pout_queue = manager.Queue()
            acc_queue = manager.Queue()
            pg_log_queue = manager.Queue()
            rs_log_queue = manager.Queue()
            pb_queue = manager.Queue()

            log_process = mp.Process(
                target=self.log_worker,
                args=(
                    self.log_dir,
                    log_level,
                    batch_id,
                    rs_log_queue,
                    pg_log_queue,
                    pb_queue,
                ),
            )
            log_process.start()
            if batch_id == 0:
                for k, v in metadata.items():
                    rs_log_queue.put((20, f"{k},  {v}"))
                    pg_log_queue.put((20, f"{k},  {v}"))
                rs_log_queue.put(
                    (
                        20,
                        f"  {'batch id':>8},  {'chn id':>8},  {'obs id':>8},  {'pin id':>8},  {'mid':^8}, {'acc':>8}",
                    )
                )
                pg_log_queue.put(
                    (
                        20,
                        f"  {'batch id':^8},  {'Channel':^8},  {'fill':^8},  {'edge':^8},  {'percep':^8},  time(ms)",
                    )
                )

            nn_process = mp.Process(
                target=self.tt_pred_worker,
                args=(obs_param_queue, tt_queue, pg_log_queue),
            )
            nn_process.start()

            acc_process = mp.Process(
                target=self.eval_worker,
                args=(self.target_fp, pout_queue, acc_queue, pg_log_queue),
            )
            acc_process.start()

            pool = mp.Pool(parallel)
            with timer:
                for chn_id in range(self.chn_bsize):
                    pool.apply_async(
                        self.compute_worker,
                        args=(
                            batch_id,
                            chn_id,
                            img_dir,
                            param_dir,
                            obs_param_queue,
                            tt_queue,
                            pout_queue,
                            acc_queue,
                            rs_log_queue,
                            pg_log_queue,
                            pb_queue,
                        ),
                    )
                pool.close()
                pool.join()
            pg_log_queue.put(
                (
                    20,
                    f"Batch {batch_id} finished. Time elapsed: {timer.get_elapsed() / 60000 : 4.1f}min",
                )
            )
            obs_param_queue.put("End")
            pout_queue.put("End")
            pg_log_queue.put("End")
            nn_process.join()
            acc_process.join()
            log_process.join()
            pg_log_queue.put((20, "Searcher complete"))

    def sample_params(self, batch_id):
        np.random.seed(1234)
        # sample obs paramters
        self.hao_coords = self.hao_coord_sampler.sample_coords(
            self.chn_bsize * self.num_hao, quasi_rand=False
        )
        self.hao_poses = np.random.randint(0, 2, size=self.chn_bsize * self.num_hao)
        hao_coord_path = os.path.join(self.log_dir, f"hao_coords_batch_{batch_id}.npy")
        hao_pos_path = os.path.join(self.log_dir, f"hao_pos_batch_{batch_id}.npy")
        np.save(hao_coord_path, self.hao_coords)
        np.save(hao_pos_path, self.hao_poses)

        # sample pin params
        fr_ratios = np.random.uniform(
            self.fr_range[0], self.fr_range[1], size=self.chn_bsize
        )
        self.pin_params = self.sample_pin(fr_ratios, self.pin_per_chn)
        # shape (num_chn, pin_per_chn, 2, c(=1))
        pin_params_path = os.path.join(self.log_dir, f"pin_params_batch_{batch_id}.npy")
        np.save(pin_params_path, self.pin_params)

    def sample_pin(self, fr_ratio, num_fp_per_chn):

        # currently only support single-color-stripe inlet profiles

        """
        fr_ratio: ndarray shape(num_chn,)
        """
        with Timer():
            _, _, w3 = self.inp_calc.calc_wr_ratio(fr_ratio, 1 - 2 * fr_ratio, fr_ratio)
            pin_l_bounds = np.linspace(
                0, 1 - w3, num_fp_per_chn
            ).T  # shape (num_chn, num_per_chn)
            pin_r_bounds = self.inp_calc.find_r_bound(
                pin_l_bounds, fr_ratio.reshape(-1, 1)
            )
            inp_bounds = (
                (np.stack([pin_l_bounds, pin_r_bounds], axis=-1) * self.fp_res[0])
                .round()
                .astype("int16")
            )
            inp_bounds = inp_bounds[
                ..., np.newaxis
            ]  # shape (num_chn, num_fp, 2, c(=1))
        return inp_bounds

    @staticmethod
    def tt_pred_worker(obs_param_queue, tt_queue, pg_log_queue):
        predictor = TTPredictor()
        while True:
            try:
                pg_log_queue.put((10, "predictor waiting input"))
                data = obs_param_queue.get(timeout=1)
                if data is None:
                    continue
                elif data == "End":
                    pg_log_queue.put((10, "predictor quitting"))
                    break
                batch_id, chn_id, obs_params = data
                pg_log_queue.put((10, f"predictor getting channel {batch_id}-{chn_id}"))
                tts = predictor.predict_from_obs_param(*obs_params)
                tt_queue.put((batch_id, chn_id, tts))
                pg_log_queue.put(
                    (10, f"predictor sent away tts chn {batch_id}-{chn_id}")
                )
            except Empty:
                continue

    @staticmethod
    def eval_worker(target_fp, pout_queue, acc_queue, pg_log_queue):
        metrics = [
            PixelwiseAccuracy(EasyDict({"op_flags": ["IoU"]})),
            PixelwiseAccuracy(EasyDict({"op_flags": ["edge", "IoU"]})),
        ]
        while True:
            try:
                pg_log_queue.put((10, "evaluator waiting input"))
                data = pout_queue.get(timeout=1)
                if data is None:
                    continue
                elif data == "End":
                    pg_log_queue.put((10, "evaluator quitting"))
                    break
                batch_id, chn_id, preds = data
                pg_log_queue.put(
                    (10, f"evaluator getting preds from channel {batch_id}-{chn_id}")
                )
                # compute the metrics
                accs = []
                for metric in metrics:
                    # next reshape pouts to BHWC to fit the metric input
                    acc = metric([preds, target_fp])  # shape (pin_per_chn*num_obs)
                    accs.append(acc)
                accs = np.stack(
                    accs, axis=0
                )  # shape (num_metrics, pin_per_chn * num_obs)
                acc_queue.put((batch_id, chn_id, accs))
                pg_log_queue.put(
                    (10, f"evaluator set away acc chn {batch_id}-{chn_id}")
                )
            except Empty:
                continue

    def log_worker(
        self, log_dir, level, batch_id, rs_log_queue, pg_log_queue, pb_queue
    ):
        result_logger = get_logger("results", log_dir, level, "a")
        progress_logger = get_logger("progress", log_dir, level, "a")
        progress_bar = tqdm(desc=f"Searching batch {batch_id}", total=self.chn_bsize, leave=False)

        while True:
            try:
                log_content = rs_log_queue.get(timeout=0.0001)
                if log_content is not None:
                    result_logger.log(*log_content)
                    log_content = None
            except Empty:
                pass
            try:
                log_content = pg_log_queue.get(timeout=0.0001)
                if log_content == "End":
                    break
                elif log_content is not None:
                    progress_logger.log(*log_content)
                    log_content = None
            except Empty:
                pass
            try:
                chn_id = pb_queue.get(timeout=0.0001)
                if chn_id is not None:
                    progress_bar.set_description(
                        f"Evaluating: Batch {batch_id} - {chn_id:>8d}"
                    )
                    progress_bar.update(1)
            except Empty:
                pass

    def compute_worker(
        self,
        batch_id,
        chn_id,
        img_dir,
        param_dir,
        obs_param_queue,
        tt_queue,
        pred_queue,
        accs_queue,
        rs_log_queue,
        pg_log_queue,
        pb_queue,
    ):
        """
        compute the cumulative transformation tensor of the given channel
        Args:
            chn_index (int): the index of the channel to be computed
        """
        pb_queue.put(chn_id)
        original_chn_id = chn_id  # debug use
        timer = Timer(verbose=False)

        with timer:
            fp_res = self.fp_res
            acc_thresholds = self.acc_thresholds
            hao_coords = self.hao_coords[
                chn_id * self.num_hao : (chn_id + 1) * self.num_hao
            ]
            hao_poses = self.hao_poses[
                chn_id * self.num_hao : (chn_id + 1) * self.num_hao
            ]
            pg_log_queue.put((10, f"computer sending param chn {batch_id}-{chn_id}"))
            obs_param_queue.put((batch_id, chn_id, (hao_coords, hao_poses)))
            batch_id, chn_id, tts = tt_queue.get()
            pg_log_queue.put((10, f"computer got tts chn {batch_id}-{chn_id}"))
            pin_params = self.pin_params[chn_id]  # shape
            cumulative_tts = tt_synth(tts, interm=True)
            pins = gen_pin_tensor(
                pin_params, fp_res, param_scale=False
            )  # shape (pin_per_chn, fp_res, fp_res, num_stripes)
            pouts = p_transform(
                pin=np.moveaxis(pins, 0, -1), tts=cumulative_tts, full_p_records=True
            )[
                1:
            ]  # shape (num_obs, fp_res, fp_res, num_stripes, pin_per_chn)
            pouts = np.moveaxis(pouts, -1, 0)
            # compute the metrics
            pg_log_queue.put(
                (10, f"computer sending away pouts chn {batch_id}-{chn_id}")
            )
            pred_queue.put(
                (
                    batch_id,
                    chn_id,
                    pouts.reshape(
                        self.pin_per_chn * self.num_hao,
                        fp_res[0],
                        fp_res[1],
                        -1,
                    ),
                )
            )

            original_chn_id = chn_id
            batch_id, chn_id, accs = accs_queue.get()
            pg_log_queue.put(
                (
                    10,
                    f"computer got accs chn {batch_id}-{chn_id}, the sent away one is {batch_id}-{original_chn_id}",
                )
            )

            accs = accs.reshape(
                -1, self.pin_per_chn, self.num_hao
            )  # shape (3, pin_per_chn, num_obs)

            acc_thresholds = np.expand_dims(acc_thresholds, (-2, -1))
            m_id, pin_id, obs_id = np.nonzero(accs > acc_thresholds)
        if len(m_id) > 0:
            fig, ax = create_profile_figure()
            for mi in range(len(m_id)):
                # plot the profiles
                pf = pouts[pin_id[mi], obs_id[mi]]
                plot_fp_tensor(pf, ax)
                # save the profile figure
                pf_fname = os.path.join(
                    img_dir,
                    f"{m_id[mi]:d}-{accs[m_id[mi],pin_id[mi],obs_id[mi]]:.2f}-{batch_id:d}-{chn_id:d}-{obs_id[mi]:d}-{pin_id[mi]:d}.png",
                )
                fig.savefig(pf_fname)
                param_fname = os.path.join(
                    param_dir,
                    f"{batch_id:d}-{chn_id:d}-{obs_id[mi]:d}-{pin_id[mi]:d}.npz",
                )
                np.savez(
                    param_fname,
                    hao_coords=hao_coords[: obs_id[mi] + 1],
                    hao_poses=hao_poses[: obs_id[mi] + 1],
                    pin_params=pin_params[pin_id[mi]],
                )
                # save the metrics
                rs_log_queue.put(
                    (
                        20,
                        f"  {batch_id:>8},  {chn_id:>8},  {obs_id[mi]:>8},  {pin_id[mi]:>8},  {m_id[mi]:>8},  {accs[m_id[mi],pin_id[mi],obs_id[mi]]:>8.3f}",
                    )
                )
            plt.close(fig)
            fill = np.count_nonzero(m_id == 0)
            edge = np.count_nonzero(m_id == 1)
            percep = np.count_nonzero(m_id == 2)
            pg_log_queue.put(
                (
                    20,
                    f"  {batch_id:>8},  {chn_id:^8},  {fill:^8},  {edge:^8},  {percep:^8},  {timer.get_elapsed()}",
                )
            )

