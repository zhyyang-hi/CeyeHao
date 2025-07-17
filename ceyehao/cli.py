#!/usr/bin/env python
"""
Command-line interface for CeyeHao package.
"""

import sys
import os
import argparse
from pathlib import Path

from ceyehao.gui.app import MainWindow
from ceyehao.config.config import parse_args, list_config, build_default_cfg
from ceyehao.utils.io import load_cfg_yml
from PyQt5.QtWidgets import QApplication


def launch_gui():
    """Launch the CeyeHao GUI application."""
    args = parse_args()
    if args.cfg_path:
        cfg = load_cfg_yml(args.cfg_path)
    else:
        print("No cfg is provided. Using built-in default config.")
        cfg = build_default_cfg()

    app = QApplication(sys.argv)
    window = MainWindow(cfg=cfg)
    print(list_config(cfg))
    window.show()
    app.exec()


def train():
   from ceyehao.train import main as train_main
   train_main()


def search():
    """Run the search script."""
    from ceyehao.search import main as search_main
    search_main()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CeyeHao: AI-driven microfluidic flow programming",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ceyehao gui                    # Launch the GUI
  ceyehao train                  # Run training
  ceyehao search                 # Run search
  ceyehao --help                 # Show this help message
        """
    )
    
    parser.add_argument(
        "command",
        choices=["gui", "train", "search"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="CeyeHao %s" % "1.0.0"
    )
    
    args = parser.parse_args()
    
    if args.command == "gui":
        launch_gui()
    elif args.command == "train":
        train()
    elif args.command == "search":
        search()


if __name__ == "__main__":
    main() 