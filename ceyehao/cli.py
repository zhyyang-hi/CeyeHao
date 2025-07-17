#!/usr/bin/env python
"""
Command-line interface for CeyeHao package.
"""

import sys
import os
import argparse
from pathlib import Path

from ceyehao.gui.app import MainWindow
from ceyehao.config.config import parse_args, list_config, build_default_cfg, build_argparser
from ceyehao.utils.io import load_cfg_yml
from PyQt5.QtWidgets import QApplication


def launch_gui(args=None):
    """Launch the CeyeHao GUI application."""
    if args is None:
        args = parse_args([])
    if getattr(args, 'cfg_path', None):
        cfg = load_cfg_yml(args.cfg_path)
    else:
        print("No cfg is provided. Using built-in default config.")
        cfg = build_default_cfg()

    app = QApplication(sys.argv)
    window = MainWindow(cfg=cfg)
    print(list_config(cfg))
    window.show()
    app.exec()


def train(args=None):
    """Run the training script."""
    from ceyehao.train import main as train_main
    train_main()


def search(args=None):
    """Run the search script."""
    from ceyehao.search import main as search_main
    search_main()


def main():
    parser = argparse.ArgumentParser(
        description="CeyeHao: AI-driven microfluidic flow programming",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-command to run")

    # GUI subparser (minimal)
    gui_parser = subparsers.add_parser("gui", help="Launch the GUI")
    gui_parser.add_argument("--cfg_path", type=str, default="", help="Path to the config YAML file for the GUI or other tasks.")
    gui_parser.set_defaults(func=launch_gui)

    # Train subparser (full config parser)
    train_parser = subparsers.add_parser("train", help="Run model training", parents=[build_argparser()], add_help=False)
    train_parser.set_defaults(func=train)

    # Search subparser (full config parser)
    search_parser = subparsers.add_parser("search", help="Run search (random or GA)", parents=[build_argparser()], add_help=False)
    search_parser.set_defaults(func=search)

    parser.add_argument(
        "--version",
        action="version",
        version="CeyeHao %s" % "1.0.0"
    )

    args = parser.parse_args()
    # Call the appropriate function
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 