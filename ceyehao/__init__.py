"""
CeyeHao: AI-driven microfluidic flow programming using hierarchically assembled obstacles 
in microchannel and receptive-field-augmented neural network

A Python package for microfluidic flow programming and design optimization.
"""

__version__ = "1.0.0"
__author__ = "Zhenyu Yang, Zhongning Jiang"
__email__ = "zhyyang@connect.hku.hk"

# Import main modules for easy access
from . import config
from . import data
from . import gui
from . import models
from . import tools
from . import utils

__all__ = [
    "config",
    "data", 
    "gui",
    "models",
    "tools",
    "utils",
] 