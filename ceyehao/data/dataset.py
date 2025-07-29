
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import random

from ceyehao.utils.data_generate import find_index


class FDataset(Dataset):
    def __init__(
        self,
        x_dir,
        y_dir,
        total_num=None,
        transform=None,
        target_transform=None,
        rand_sym_ratio=0,
        shuffle=True,
    ):
        self.x_dir = x_dir
        self.y_dir = y_dir
        self.total_num = total_num if total_num else len(os.listdir(x_dir))

        x_list = os.listdir(x_dir)
        y_list = os.listdir(y_dir)
        try:
            x_list.remove(".ipynb_checkpoints")
            y_list.remove(".ipynb_checkpoints")
        except:
            pass
        x_list.sort(key=find_index)
        y_list.sort(key=find_index)
        self.xy_list = list(zip(x_list, y_list))
        if shuffle:
            random.shuffle(self.xy_list)
        self.xy_list = self.xy_list[: self.total_num]

        self.transform = transform
        self.target_transform = target_transform
        self.rand_sym_ratio = rand_sym_ratio
        if self.rand_sym_ratio > 0:
            assert self.target_transform
            assert self.transform

    def __len__(self):
        return len(self.xy_list)

    def __getitem__(self, idx):
        x_name, y_name = self.xy_list[idx]

        x = cv2.imread(os.path.join(self.x_dir, x_name), cv2.IMREAD_GRAYSCALE)
        y = np.load(os.path.join(self.y_dir, y_name))
        if self.rand_sym_ratio > torch.rand(1):
            sym_trigger = True
        else:
            sym_trigger = False
        if self.transform:
            x = self.transform(x, sym_trigger)
        if self.target_transform:
            y = self.target_transform(y, sym_trigger)
        return x, y.astype(np.float32).transpose(2, 0, 1)
