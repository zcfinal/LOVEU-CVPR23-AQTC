## A solution to the CVPR'23 LOVEU-AQTC challenge 

### Video Alignment for Multi-step Inference

This repo provides the **2nd place** solution of the CVPR'23 LOVEU-AQTC challenge.

![image](https://github.com/zcfinal/LOVEU-CVPR23-AQTC/assets/48513057/33a66e28-cec1-46de-9b64-430eccda5c36)


[[Challenge Page]](https://showlab.github.io/assistq/)  [[Challenge Paper]](https://arxiv.org/abs/2203.04203) [[LOVEU@CVPR'23 Challenge]](https://sites.google.com/view/loveucvpr23/track3?authuser=0) [[CodaLab Leaderboard]](https://codalab.lisn.upsaclay.fr/competitions/4642#results)

Click to know the task:

[![Click to see the demo](https://img.youtube.com/vi/3v8ceel9Mos/0.jpg)](https://www.youtube.com/watch?v=3v8ceel9Mos)




## Install

(1) PyTorch. See https://pytorch.org/ for instruction. For example,

```
conda install pytorch torchvision torchtext cudatoolkit=11.3 -c pytorch
```

(2) PyTorch Lightning. See https://www.pytorchlightning.ai/ for instruction. For example,

```
pip install pytorch-lightning
```

(3) [VideoCLIP install](https://github.com/facebookresearch/fairseq/tree/main/examples/MMPT)


## Data

Download training set and testing set (without ground-truth labels) of CVPR'22 LOVEU-AQTC challenge by filling in the [[AssistQ Downloading Agreement]](https://forms.gle/h9A8GxHksWJfPByf7).

Then carefully set your data path in the config file ;)

## Encoding

We utilize pretrained [S3D](https://github.com/antoine77340/S3D_HowTo100M) and [VideoCLIP](https://github.com/facebookresearch/fairseq/tree/main/examples/MMPT) to encode the videos and scripts.

/pretrain/feature.sh is the script to conduct encoding.

## Training & Evaluation

/sh/search_dim.sh is the training script.

/sh/search_inf.sh is the inference script.

ensemble_b.py and ensemble.py are the file to ensemble results.
