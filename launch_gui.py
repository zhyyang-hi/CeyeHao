#!/usr/bin/env python

import sys, os
from ceyehao.gui.app import MainWindow
from PyQt5.QtWidgets import QApplication
from ceyehao.config.config import parse_args, list_config
from ceyehao.utils.io import load_cfg_yml

args = parse_args()
if args.cfg_path:
    cfg = load_cfg_yml(args.cfg_path)
else:
    print("No cfg is provided. Loading default CEyeNet.")
    cfg = load_cfg_yml(r"../log/CEyeNet/infer_cfg.yml")

app = QApplication(sys.argv)
window = MainWindow(cfg=cfg)
print(list_config(cfg))
window.show()
app.exec()
