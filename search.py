from tools.inverse_design import MirochannelDesignSearcher
from config.config import parse_args, create_cfg_from_args
from utils.io import load_cfg_yml

args = parse_args()
cfg = load_cfg_yml(args.cfg_path)
searcher = MirochannelDesignSearcher(cfg)
searcher()