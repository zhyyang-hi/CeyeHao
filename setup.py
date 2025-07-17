from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ceyehao",
    version="1.0.0",
    author="Zhenyu Yang, Zhongning Jiang, Haisong Lin, Edmund Y. Lam, Hayden Kwok-Hay So, Ho Cheung Shum",
    author_email="elam@eee.hku.hk, hso@eee.hku.hk, ashum@cityu.edu.hk",
    description="AI-driven microfluidic flow programming using hierarchically assembled obstacles in microchannel and receptive-field-augmented neural network",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/zhyyang-hi/CeyeHao",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "ceyehao=ceyehao.cli:main",
            "ceyehao-gui=ceyehao.cli:launch_gui",
            "ceyehao-train=ceyehao.cli:train",
            "ceyehao-search=ceyehao.cli:search",
        ],
    },
    include_package_data=True,
    package_data={
        "ceyehao": ["config/*.yml", "config/*.pkl", "gui/*.ui"],
    },
) 