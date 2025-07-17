import os, sys

parent_path = os.path.abspath(os.path.join(__file__, *([".."] * 2)))
sys.path.insert(0, parent_path)
import glob
import pickle
from tqdm import tqdm
import torch
import numpy as np
import matplotlib
import pandas as pd

matplotlib.use("agg")
import matplotlib.pyplot as plt
from ceyehao.tools.trainer import Trainer
from ceyehao.utils.utils import mkdirs, Timer
from ceyehao.utils.io import dump_cfg_yml
from ceyehao.utils.data_process import tt_postprocess
from ceyehao.utils.visualization import (
    plot_tt_contour,
    plot_tt_contour_colormap,
    create_profile_figure,
)


def list_models(root_dir, to_file=True):
    """
    list all the model paths in the root_dir
    """
    model_paths = [
        file_dir
        for file_dir in glob.iglob(root_dir + "**/**", recursive=True)
        if os.path.isfile(file_dir)
        and (
            os.path.basename(file_dir).startswith("UNet")
            or os.path.basename(file_dir).startswith("CEyeNet")
            or os.path.basename(file_dir).startswith("GVTN")
            or os.path.basename(file_dir).startswith("AsymUNet")
        )
        and not os.path.basename(file_dir).endswith("_last")
    ]

    model_paths = sorted(list(set(model_paths)))

    if to_file:
        with open(os.path.join(root_dir, "model_path_list.txt"), "w") as f:
            f.write("\n".join(model_paths))

    return model_paths


class ModelEvaluator(Trainer):
    def __init__(self, cfg):
        super().__init__(cfg)

    def eval(self):
        eval_programs = self.cfg.eval_cfg.programs
        exports = self.cfg.eval_cfg.exports
        eval_results = {}
        timer = Timer(verbose=False)
        infer_time_records = []
        postprcs_t_records = []
        preds_dir = mkdirs(os.path.join(self.cfg.eval_cfg.result_dir, "preds"))

        progress_bar = tqdm(
            enumerate(self.valid_loader),
            leave=False,
            total=len(self.valid_loader),
            desc="inferring transformation tensors",
        )
        for i, data in progress_bar:
            # batch pred
            inputs, labels = data
            inputs, labels = inputs.to(self.cfg.device), labels.to(self.cfg.device)

            with timer:
                with torch.no_grad():
                    with torch.autocast(
                        device_type=self.cfg.device,
                        dtype=torch.float16,
                        enabled=self.cfg.amp,
                    ):
                        outputs = self.model(inputs)
                        if self.cfg.model == "gvtn":
                            outputs = outputs[1]
            infer_time_records.append(timer.get_elapsed())

            with timer:
                outputs = tt_postprocess(outputs)
            postprcs_t_records.append(timer.get_elapsed())
            np.save(os.path.join(preds_dir, f"preds_batch_{i}.npy"), outputs)

        if eval_programs.accuracy:
            accuracies = []
            acc_metric = self.create_acc_metric(self.cfg)
            progress_bar.set_description("evaluating accuracy")

            progress_bar = tqdm(
                enumerate(self.valid_loader),
                leave=False,
                total=len(self.valid_loader),
                desc="evaluating accuracy",
            )
            for i, (_, labels) in progress_bar:
                labels = tt_postprocess(labels)
                pred_batch = np.load(os.path.join(preds_dir, f"preds_batch_{i}.npy"))
                batch_acc = acc_metric([pred_batch, labels])
                accuracies.append(batch_acc)
            accuracies = np.concatenate(accuracies)

            idx = np.arange(0, accuracies.shape[0]).reshape(-1, 1).astype("int16")
            acc_table = pd.DataFrame(
                np.concatenate([idx, accuracies], axis=-1), columns=["idx", "acc"]
            )
            print('\nAccuracy statistics:\n', acc_table.describe())
            eval_results["accuracies"] = acc_table

        if eval_programs.time:
            infer_time_records = np.array(infer_time_records)
            postprcs_t_records = np.array(postprcs_t_records)
            dataset_size = len(self.valid_loader.dataset)
            batch_num = len(self.valid_loader)
            last_batch_size = dataset_size % self.cfg.data_cfg.valid_bs
            batch_sizes = [self.cfg.data_cfg.valid_bs] * (batch_num - 1) + [
                last_batch_size
            ]
            time_table = pd.DataFrame(
                np.stack([batch_sizes, infer_time_records, postprcs_t_records], axis=1),
                columns=["batch_size", "infer_time", "postprocess_time"],
            )
            eval_results["time"] = time_table
            print('\Time statistics:\n', time_table.describe())

        if exports.eval_results:
            for key, value in eval_results.items():
                value.to_csv(
                    os.path.join(self.cfg.eval_cfg.result_dir, f"{key}.csv"),
                    index=False,
                )
        if exports.cfg:
            dump_cfg_yml(
                self.cfg, os.path.join(self.cfg.eval_cfg.result_dir, "cfg.yml")
            )
        if exports.model_checkpoint:
            torch.save(
                self.model.state_dict(),
                os.path.join(self.cfg.eval_cfg.result_dir, "model.pt"),
            )

        if exports.plot.tt_pred:
            pred_id = 0
            figure, ax = create_profile_figure()
            plot_dir = os.path.join(self.result_path, "preds_plots")

            pred_paths = glob.glob(os.path.join(preds_dir, "*.npy"))
            for i in tqdm(range(len(pred_paths)), desc="plot pred contour"):
                pred_batch = np.load(os.path.join(preds_dir, f"preds_batch_{i}.npy"))
                for j in range(pred_batch.shape[0]):
                    plot_tt_contour(pred_batch[j], ax)
                    figure.savefig(os.path.join(plot_dir, f"pred_{pred_id}.png"))
                    pred_id += 1
            plt.close(figure)

        if exports.plot.tt_label:
            label_id = 0
            figure, ax = create_profile_figure()
            for _, labels in tqdm(
                self.valid_loader,
                desc="plot label contour",
                total=len(self.valid_loader),
            ):
                labels = tt_postprocess(labels)
                for j in range(labels.shape[0]):
                    plot_tt_contour(labels[j], ax)
                    figure.savefig(os.path.join(plot_dir, f"label_{label_id}.png"))
                    label_id += 1
            plt.close(figure)
