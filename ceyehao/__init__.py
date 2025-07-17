"""
CeyeHao: AI-driven microfluidic flow programming using hierarchically assembled obstacles 
in microchannel and receptive-field-augmented neural network

A Python package for microfluidic flow programming and design optimization.
"""

__version__ = "1.0.0"
__author__ = "Zhenyu Yang, Zhongning Jiang, Haisong Lin, Edmund Y. Lam, Hayden Kwok-Hay So, Ho Cheung Shum"
__email__ = "elam@eee.hku.hk, hso@eee.hku.hk, ashum@cityu.edu.hk"

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