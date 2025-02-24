import os
import shutil
import logging
from datetime import datetime
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from timeit import default_timer

matplotlib.use("Agg")


def plot_line(train_x, train_y, valid_x, valid_y, mode, out_dir):
    """plot loss / acc curve during training

    Args:
        train_x : epoch num range -> x axis of trian figure
        train_y : y axis of train figure
        valid_x : epoch num range -> x axis of valid figure
        valid_y : y axis of valid figure
        mode(str) : 'loss' or 'acc'
        out_dir : save path of the figure
    """
    plt.plot(train_x, train_y, label="Train")
    plt.plot(valid_x, valid_y, label="Valid")

    plt.ylabel(str(mode))
    plt.xlabel("Epoch")

    location = "upper right" if mode == "loss" else "upper left"
    plt.legend(loc=location)

    plt.title("_".join([mode]))
    plt.savefig(os.path.join(out_dir, mode + ".png"))
    plt.close()


def mkdirs(path: str, remove_old: bool = False):
    """make directory if not exist. 
    Args:
        path : path to make directory
        remove_old : if True, remove old directory and make new one. if False, keep old directory as it is.
    Returns:
        path : path of the directory
    """
    if os.path.exists(path):
        if remove_old:
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)
    return path


def generate_datetime_folder_tree(exp_path):
    """create folder tree
    exp_path:
    |- YYYY_MM_DD
        |- HH_MM_SS
    """
    now = datetime.now()
    folder_name = now.strftime("%H_%M_%S")
    date = now.strftime("%Y_%m_%d")
    result_save_folder = os.path.join(exp_path, date, folder_name)
    if not os.path.exists(result_save_folder):
        os.makedirs(result_save_folder)
    return result_save_folder


class Logger(object):
    def __init__(self, path_log):
        log_name = os.path.basename(path_log)
        self.log_name = log_name if log_name else "root"
        self.out_path = path_log

        log_dir = os.path.dirname(self.out_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def init_logger(self):
        logger = logging.getLogger(self.log_name)
        logger.setLevel(level=logging.INFO)

        # File Handler
        file_handler = logging.FileHandler(self.out_path, "w")
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)

        # Screen Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

        # add handler
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger


def create_logger(log_root="./log"):
    log_dir = generate_datetime_folder_tree(log_root)
    path_log = os.path.join(log_dir, "log.log")
    logger = Logger(path_log)
    logger = logger.init_logger()
    return logger, log_dir


class Timer(object):
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.timer = default_timer
        self.elapsed = None

    def __enter__(self):
        self.start = self.timer()
        return self

    def __exit__(self, *args):
        end = self.timer()
        self.elapsed_secs = end - self.start
        self.elapsed = self.elapsed_secs * 1000  # millisecs
        if self.verbose:
            tqdm.write("elapsed time: %.3f ms" % self.elapsed)

    def get_elapsed(self):
        return self.elapsed.__round__(1)
