<h1 align="center">CeyeHao: AI-driven microfluidic flow programming using hierarchically assembled obstacles in microchannel and receptive-field-augmented neural network</h1>
<h4 align="center"><a href="https://doi.org/10.5281/zenodo.13363708"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.13363708.svg" alt="DOI"></a></h4>
</h4>
<div align="center">
Zhenyu Yang<sup>1,2,8</sup>, Zhongning Jiang<sup>3,8</sup>, Haisong Lin<sup>5,6</sup>, Edmund Y. Lam<sup>7,9</sup>, Hayden Kwok-Hay So<sup>7,9</sup>, Ho Cheung Shum<sup>1,2,3,4,9</sup>
</div>
<div align="center">
  <sup>1</sup>Advanced Biomedical Instrumentation Centre, Hong Kong, China. <br>
  <sup>2</sup>Department of Mechanical Engineering, The University of Hong Kong, Hong Kong, China.<br>
  <sup>3</sup>Department of Biomedical Engineering, City University of Hong Kong, Hong Kong, China.<br>
  <sup>4</sup>Department of Biomedical Engineering, City University of Hong Kong, Hong Kong, China.<br>
  <sup>5</sup>School of Engineering, Westlake University, Hangzhou, China.<br>
  <sup>6</sup>Research Center for Industries of the Future, Westlake <br>University, Hangzhou, China.
  <sup>7</sup>Department of Electrical and Electronic Engineering, The University of Hong Kong, Hong Kong, China.<br>
  <sup>8</sup>These authors contributed equally: Zhenyu Yang, Zhongning Jiang.<br>
  <sup>9</sup>These authors are the corresponding authors. e-mail: elam@eee.hku.hk, hso@eee.hku.hk, ashum@cityu.edu.hk.
</div>

## Introduction and setup
This is the code implementations for the manuscript titled "CeyeHao: AI-driven microfluidic flow programming using hierarchically assembled obstacles in microchannel and receptive-field-augmented neural network"

To conduct similar studies as those presented in the manuscript, start by cloning this repository via
```
git clone https://github.com/zhyyang-hi/CeyeHao.git
```

The dataset for model training and model checkpoint are provided on [Zenodo](https://zenodo.org/records/13363708). Unzip the dataset `dataset.zip` into the `../dataset` folder and the pre-trained model checkpoint `checkpoint.zip` in the `../log` folder. To automatic search the microchannel design for a specific output flow profile, put the flow porfile image into `../auto_search/target_profile.png`.

The complete directory tree of the codes and data is shown below. 
```
..
├── dataset
│   ├── obs_img_train
│   │   └── ...
│   ├── obs_img_valid
│   │   └── ...
│   ├── tt_train
│   │   └── ...
│   └── tt_valid
│       └── ...
├── log
|   └── CEyeNet
|       ├── CEyeNet
|       └── ...
├── auto_search
|   └── target_profile.png
└── FlowProgrammer
    └── ...
```

To run the codes, first install the dependent packages described in `./requirements.txt`

To train a CEyeNet model, run
```
python train.py
```
To launch the GUI for heuristic design with the pretrained CEyeNet Checkpoint, run
```
python launch_gui.py
``` 
To automatically search the microchannel design, run
```
python search.py
```


Configurations of traning, evaluation, and inferring are listed in `./config/template.yml` 

For more details, please refer to the manuscript. 
