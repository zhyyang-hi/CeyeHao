# from tools.inverse_design import MirochannelDesignSearcher
# from config.config import parse_args, create_cfg_from_args
# from utils.io import load_cfg_yml

# args = parse_args()
# cfg = load_cfg_yml(args.cfg_path)
# searcher = MirochannelDesignSearcher(cfg)
# searcher()

import torch.multiprocessing as mp
from ceyehao.tools.inverse_design import InvDesign
import cv2
from ceyehao.config.config import parse_args
from ceyehao.utils.utils import mkdirs


def main():
    args = parse_args()
    LOG_DIR = mkdirs(args.log_dir, remove_old=True)
    TGT_PTH = args.tgt_pth
    NUM_CHN = args.num_chn
    NUM_HAO = args.num_hao
    NUM_PIN = args.num_pin

    mp.set_start_method("spawn")
    tgt_img = cv2.imread(TGT_PTH)
    tgt_img = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2GRAY)
    tgt_img = cv2.resize(tgt_img, (200, 200))

    searcher = InvDesign()
    searcher.set_task(
        tgt_fp              =   tgt_img,
        num_chn             =   NUM_CHN,
        chn_batch_size      =   2000,
        num_hao             =   NUM_HAO,
        pin_per_chn         =   NUM_PIN,
        restrict_flow_ratio =   True,
        acc_thresholds      =   [0.6, 0.3],
        log_dir             =   LOG_DIR,
    )
    searcher.run_task(8, log_level=20)


if __name__ == "__main__":
    main()
