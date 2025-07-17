import os
from typing import *
import pickle
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.dataset import FDataset
from data.transform import TransformObsImg, TransformTT

from models.archs import *
from models.unetpp import Generic_UNetPlusPlus as Unetpp
from models.gvtn.network import GVTN
from tools.loss_functions import LossFunc
from tools.scheduler import CosineWarmupLr
from tools.acc_metrics import *
from utils.utils import create_logger, plot_line
from utils.io import dump_cfg_yml
from config.config import MODE, SUPPORTED_MODELS, list_config

# manual seedings
# random.seed(0)
# np.random.seed(0)
# torch.manual_seed(0)


class Trainer:
    def __init__(self, cfg):
        assert cfg.mode in MODE, f"mode should be train/eval/infer, not '{cfg.mode}' ."
        assert cfg.model in SUPPORTED_MODELS, f"model '{cfg.model}' is not supported."
        if not cfg.device:
            cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cfg = cfg

        if cfg.mode == "train":
            self.logger, self.log_dir = self.create_logger(self.cfg)
        self.build_model()
        self.scaler = self.create_gradscaler(self.cfg)  # amp
        if cfg.mode in ["train", "eval"]:
            self.train_loader, self.valid_loader = self.create_loader()
        if cfg.mode == "train":
            # self.export_model(self.model, os.path.join(self.log_dir, "model.pt"))
            self.loss_f = self.create_loss_f(self.cfg)
            self.optimizer = self.create_optimizer(self.cfg)
            self.scheduler = self.create_scheduler(self.cfg)
            self.acc_metric = self.create_acc_metric(self.cfg)

    def train(self):
        with logging_redirect_tqdm([self.logger]):
            self.logger.info("Config:\n" + list_config(self.cfg))
        # export cfg with pickle
        dump_cfg_yml(self.cfg, os.path.join(self.log_dir, "cfg.yml"))
        with open(os.path.join(self.log_dir, "cfg.pickle"), "wb") as p:
            pickle.dump(self.cfg, p)

        # log model trainable paramters
        with logging_redirect_tqdm([self.logger]):
            self.logger.info(
                "Trainable parameters: {}".format(self.get_number_of_parameters())
            )
        loss_rec = {"train": [], "val": []}
        acc_rec = {"train": [], "val": []}
        best_acc = 0
        best_epoch = 0
        model_name = self.model.__class__.__name__

        for epoch in tqdm(
            range(self.cfg.train_cfg.max_epoch), desc="training...", leave=False
        ):
            # train
            loss_train, acc_train = self.train_epoch(
                data_loader=self.train_loader,
                model=self.model,
                loss_f=self.loss_f,
                optimizer=self.optimizer,
                cfg=self.cfg,
                epoch_idx=epoch,
                logger=self.logger,
                acc_metric=self.acc_metric,
                scheduler=self.scheduler,
                scaler=self.scaler,
            )
            # val
            loss_val, acc_val = self.valid_epoch(
                data_loader=self.valid_loader,
                model=self.model,
                loss_f=self.loss_f,
                acc_metric=self.acc_metric,
                cfg=self.cfg,
            )

            self.scheduler.step()

            with logging_redirect_tqdm([self.logger]):
                self.logger.info(
                    "Epoch[{:0>3}/{:0>3}] \t Train loss: {:.6f} \t Train Acc: {:.4f} \t Valid Acc:{:.4f} \t LR:{} \n".format(
                        epoch,
                        self.cfg.train_cfg.max_epoch,
                        loss_train,
                        acc_train,
                        acc_val,
                        self.optimizer.param_groups[0]["lr"],
                    )
                )

            # record train info
            loss_rec["train"].append(loss_train), loss_rec["val"].append(loss_val)
            acc_rec["train"].append(acc_train), acc_rec["val"].append(acc_val)

            # visualization
            plt_x = np.arange(1, epoch + 2)
            plot_line(
                plt_x,
                loss_rec["train"],
                plt_x,
                loss_rec["val"],
                mode="loss",
                out_dir=self.log_dir,
            )
            plot_line(
                plt_x,
                acc_rec["train"],
                plt_x,
                acc_rec["val"],
                mode="acc",
                out_dir=self.log_dir,
            )

            # save model
            if best_acc < acc_val or epoch == self.cfg.train_cfg.max_epoch - 1:
                if best_acc < acc_val:
                    model_name = self.model.__class__.__name__
                    best_epoch = epoch
                    best_acc = acc_val
                else:
                    model_name = self.model.__class__.__name__ + "_last"
                    best_epoch = best_epoch
                    best_acc = best_acc

                save_path = os.path.join(self.log_dir, model_name)
                torch.save(self.model.state_dict(), save_path)
                with logging_redirect_tqdm([self.logger]):
                    self.logger.info(
                        "Best in Epoch {}, acc: {:.4f}".format(best_epoch, best_acc)
                    )

        # finish
        with logging_redirect_tqdm([self.logger]):
            self.logger.info(
                "{} trianing done, best acc: {:.4f}, in Epoch {}".format(
                    model_name, best_acc, best_epoch
                )
            )
        self.logger.handlers.clear()
        self.logger = None

    @staticmethod
    def create_dataset(cfg):
        root_dir = cfg.data_cfg.data_root_dir
        tf_x = TransformObsImg(cfg)
        tf_y = TransformTT(cfg)

        if hasattr(
            cfg.data_cfg, "dataset_size"
        ) is False or cfg.data_cfg.dataset_size in [None, [9000, 1000]]:
            train_x_dir = os.path.join(root_dir, "obs_imgs_train/")
            train_y_dir = os.path.join(root_dir, f"tt{cfg.profile_size[0]}_train/")
            valid_x_dir = os.path.join(root_dir, "obs_imgs_valid/")
            valid_y_dir = os.path.join(root_dir, f"tt{cfg.profile_size[0]}_valid/")
            train_set = FDataset(
                train_x_dir,
                train_y_dir,
                transform=tf_x,
                target_transform=tf_y,
                shuffle=True,
            )
            valid_set = FDataset(
                valid_x_dir,
                valid_y_dir,
                transform=tf_x,
                target_transform=tf_y,
                shuffle=False,
            )
        else:
            dataset = FDataset(
                os.path.join(root_dir, "obs_imgs/"),
                os.path.join(root_dir, f"tt{cfg.profile_size[0]}/"),
                transform=tf_x,
                target_transform=tf_y,
                total_num=sum(cfg.data_cfg.dataset_size),
            )
            train_set, valid_set = torch.utils.data.random_split(
                dataset, cfg.data_cfg.dataset_size
            )
        tqdm.write("Dataset created.")
        return train_set, valid_set

    def create_loader(self):
        train_set, valid_set = self.create_dataset(self.cfg)
        train_loader = DataLoader(
            train_set,
            batch_size=self.cfg.data_cfg.train_bs,
            shuffle=True,
            num_workers=self.cfg.data_cfg.workers,
        )
        valid_loader = DataLoader(
            valid_set,
            batch_size=self.cfg.data_cfg.valid_bs,
            num_workers=self.cfg.data_cfg.workers,
        )

        tqdm.write("Data loader created.")

        return train_loader, valid_loader

    def build_model(self):
        model = self.cfg.model
        model_cfg = self.cfg.model_cfg
        profile_size = self.cfg.profile_size

        if model == "UNet":
            model = UNet(output_size=profile_size, **model_cfg)
        elif model == "CEyeNet":
            model = CEyeNet(output_size=profile_size, **model_cfg)
        elif model == "UNet++":
            model = Unetpp(**model_cfg)
        elif model == "AsymUNet":
            model = AsymUNet(
                output_size=profile_size,
                **model_cfg,
            )
        elif model == "gvtn":
            model = GVTN(
                **model_cfg,
            )
        else:
            raise RuntimeError("Unimplemented Model!")

        if self.cfg.model_checkpoint:
            model.load_state_dict(
                torch.load(self.cfg.model_checkpoint, map_location="cpu")
            )
            # if self.cfg.mode == "train":
            #     with logging_redirect_tqdm([self.logger]):
            #         self.logger.info(
            #             f"Pretrained model is loaded from {self.cfg.model_checkpoint}"
            #         )
            # else:
            #     tqdm.write(f"Loaded model from {self.cfg.model_checkpoint}")
            #     model.eval()
        model.to(self.cfg.device)
        # tqdm.write("Model built.")
        self.model = model
        return model

    def create_logger(self, cfg):
        logger, log_dir = create_logger(cfg.train_cfg.log_dir)
        tqdm.write("Logger created.")
        return logger, log_dir

    def create_optimizer(self, cfg):
        optimizer = optim.SGD(
            params=self.model.parameters(),
            weight_decay=cfg.train_cfg.weight_decay,
            lr=cfg.train_cfg.lr_init,
            momentum=cfg.train_cfg.momentum,
        )
        tqdm.write("Optimizer created.")
        return optimizer

    def create_scheduler(self, cfg):
        if cfg.train_cfg.is_warmup:
            iter_per_epoch = len(self.train_loader)
            scheduler = CosineWarmupLr(
                self.optimizer,
                batches=iter_per_epoch,
                max_epochs=cfg.train_cfg.max_epoch,
                base_lr=cfg.train_cfg.lr_init,
                final_lr=cfg.train_cfg.lr_final,
                warmup_epochs=cfg.train_cfg.warmup_epochs,
                warmup_init_lr=cfg.train_cfg.lr_warmup_init,
            )
        else:
            scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                gamma=cfg.train_cfg.factor,
                milestones=cfg.train_cfg.milestones,
            )
        tqdm.write("Scheduler created.")
        return scheduler

    def create_loss_f(self, cfg):
        loss_f = LossFunc(cfg)
        tqdm.write("Loss function created.")
        return loss_f

    @staticmethod
    def create_acc_metric(cfg):
        acc_metric = PixelwiseAccuracy(cfg.pix_acc_cfg)
        tqdm.write("Accuracy metric created.")
        return acc_metric

    def create_gradscaler(self, cfg):
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)
        return scaler

    @staticmethod
    def train_epoch(
        data_loader: DataLoader,
        model: nn.Module,
        loss_f,
        optimizer,
        cfg,
        epoch_idx,
        logger,
        acc_metric,
        scheduler=None,
        scaler=None,
    ):
        model.train()
        loss_sigma = []
        acc_accum = 0
        sample_count = 0  # total sample count of each epoch
        for i, data in tqdm(
            enumerate(data_loader),
            desc="train batch",
            leave=False,
            total=len(data_loader),
        ):
            inputs, labels = data
            inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)
            with torch.autocast(
                device_type=cfg.device, dtype=torch.float16, enabled=cfg.amp
            ):
                # forward
                outputs = model(inputs)
                # backward
                loss = loss_f(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            # acc and loss cal
            if cfg.model == "gvtn":
                outputs = outputs[1]  # unpack gtvn output and only keep the prediction
            acc_accum += acc_metric(
                [outputs.detach().cpu().numpy(), labels.detach().cpu().numpy()]
            ).sum()
            loss_sigma.append(loss.item())
            sample_count += len(inputs)

            # print train info by interval
            if i % cfg.train_cfg.log_interval == cfg.train_cfg.log_interval - 1:
                with logging_redirect_tqdm([logger]):
                    logger.info(
                        "|Epoch[{}/{}]||batch[{}/{}]||batch_loss: {:.6f}||accuracy: {:.4f}|".format(
                            epoch_idx,
                            cfg.train_cfg.max_epoch,
                            i + 1,
                            len(data_loader),
                            loss.item(),
                            float(acc_accum / sample_count),
                        )
                    )
        # cal mean acc and loss
        loss_mean = np.mean(loss_sigma)  # mean loss of each epoch
        acc_mean = np.mean(float(acc_accum / sample_count))  # mean acc of each epoch

        return loss_mean, acc_mean

    @staticmethod
    def valid_epoch(
        data_loader: DataLoader,
        model: nn.Module,
        loss_f,
        cfg,
        acc_metric,
    ):
        model.eval()
        loss_sigma = []
        val_acc_accum = 0
        sample_count = 0  # total sample count of valid epoch
        for i, data in tqdm(
            enumerate(data_loader),
            desc="eval batch",
            leave=False,
            total=len(data_loader),
        ):
            inputs, labels = data
            inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)
            with torch.no_grad():
                with torch.autocast(
                    device_type=cfg.device, dtype=torch.float16, enabled=cfg.amp
                ):
                    # forward
                    outputs = model(inputs)
                    loss = loss_f(outputs, labels)
            # accum loss
            loss_sigma.append(loss.item())
            # evaluate
            if cfg.model == "gvtn":
                outputs = outputs[1]
            acc_batch = acc_metric(
                [outputs.detach().cpu().numpy(), labels.detach().cpu().numpy()]
            ).sum()
            val_acc_accum += acc_batch
            sample_count += len(inputs)
        # tqdm.write("\n")
        loss_mean = np.mean(loss_sigma)
        val_acc_mean = np.mean(float(val_acc_accum / sample_count))
        return loss_mean, val_acc_mean

    @staticmethod
    def export_model(model, fname):
        # get current model mode
        mode = model.training
        if mode:
            model.eval()
        x = torch.randn(1, 1, 200, 200).to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        # torch.jit.trace(model, x).save(fname)
        # restore model mode
        if mode:
            model.train()

    def get_number_of_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
