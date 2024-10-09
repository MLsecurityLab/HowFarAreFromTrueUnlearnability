# How Far Are We From True Unlearnability
[![Python](https://img.shields.io/badge/Python-3.9-green)](https://www.python.org/downloads/release/python-3919/)
[![PyTorch](https://img.shields.io/badge/PyTorch-0.9.1-green)](https://pytorch.org/)
[![MIT](https://img.shields.io/badge/License-MIT-green)](https://github.com/lafeat/apbench/blob/main/LICENSE)

High-quality data plays an indispensable role in the era of large models, but the use of unauthorized data for model training greatly damages the interests of data owners. To overcome this threat, several unlearnable methods have been proposed, which generate unlearnable examples (UEs) by compromising the training availability of data. Clearly, due to unknown training purpose and the powerful representation learning capabilities of existing models, these data are expected to be unlearnable for various task models, i.e., they will not help improve the model's performance. However, unexpectedly, we find that on the multi-task dataset Taskonomy, UEs still perform well in tasks such as semantic segmentation, failing to exhibit cross-task unlearnability. This phenomenon leads us to question: How far are we from attaining truly unlearnable examples? We attempt to answer this question from the perspective of model optimization. We observe the difference of convergence process between clean models and poisoned models on a simple model using the loss landscape and find that only a part of the critical parameter optimization paths show significant differences, implying a close relationship between the loss landscape and unlearnability. Consequently, we employ the loss landscape to explain the underlying reasons for UEs and propose Sharpness-Aware Learnability (SAL) for quantifying the unlearnability of parameters based on this explanation. Furthermore, we propose an Unlearnable Distance (UD) metric to measure the unlearnability of data based on the SAL distribution of parameters in clean and poisoned models. Finally, we conduct benchmark tests on mainstream unlearnable methods using the proposed UD, aiming to promote community awareness of the capability boundaries of existing unlearnable methods. 

### [Project Page](TODO) | [Paper](TODO)

> How Far Are We From True Unlearnability

![teaser](assets/teaser.gif)


## Citation
If you find this paper helpful for your research, please cite:
```bibtex
@inproceedings{
anonymous2024how,
title={How Far Are We from True Unlearnability?},
author={Anonymous},
booktitle={Submitted to The Thirteenth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=I4Lq2RJ0eJ},
note={under review}
}
```

## News
📢 **1/Oct/24** - First release


## Table of Contents

* [Overview](#Overview)
* [Installation](#Installation)
* [Code Structure](#code-structure)
* [Quick Start](#quick-start)
* [Supported Methods](#supported-methods)
* [Unsupervised Methods](#unsupervised-methods)

## Overview

The experiment of this paper contains the following attacks and defenses related to Availability Attack (Unlearnable Examples):

**Attacks**:
  - 6 availability attack methods:
  [DeepConfuse](https://papers.nips.cc/paper_files/paper/2019/file/1ce83e5d4135b07c0b82afffbe2b3436-Paper.pdf),
  [EM](https://openreview.net/pdf?id=iAmZUo0DxC0),
  [REM](https://openreview.net/pdf?id=baUQQPwQiAg),
  [TAP](https://arxiv.org/pdf/2106.10807.pdf),
  [LSP](https://arxiv.org/pdf/2111.00898.pdf),
  [OPS](https://arxiv.org/pdf/2205.12141.pdf),

  
**Defenses**: 
  - 3 availability poisoning defense methods:
  [AT](https://arxiv.org/pdf/1706.06083v2.pdf),
  [ISS](https://arxiv.org/pdf/2301.13838.pdf),
  [UEraser](https://arxiv.org/pdf/2303.15127.pdf).
  
**Datasets**: CIFAR-10, CIFAR-100.
 
**Models**: ResNet-18, ResNet-50, SENet-18, Vit-small.
 
## Getting started
You can run the following script to configurate necessary environment:

```bash
conda create -n UD python=3.9
conda activate UD
pip install -r requirements.txt


## Download dataset
Please download CIFAR-10 and CIFAR-100 to get poisoned dataset yourself, and put them into `dataset\`. We recommend using the [benchmark](https://openreview.net/pdf?id=igJ2XPNYbJ) to generate availability attack related data.

## Quick Start

**Step 1: Sharpness-Aware Learning (SAL) on poisoned datasets**: 
If you have already generated poisoned dataset, you can train the model with a demo script below:
```bash
python train_sharp.py --dataset <Dataset> --<Defense> --arch <Model_arch> --type <Attack>
```
The parameter choices for the above commands are as follows:
- --dataset `<Dataset>`: `c10` , `c100`.
- --`<Defense>`: `nodefense`, `cutout`, `cutmix`, `mixup`, `mixup`, `bdr`, `gray`, `jpeg`, `gaussian`, `ueraser`, `at`.
- --arch `<Model_arch>`: `r18`, `r50`, `se18`, `vit`.
- --type `<Attack>`: `ar`, `dc`, `em`, `rem`, `hypo`, `tap`, `lsp`, `ntga`, `ops`.
  
The trained checkpoints will be saved at `log/<Dataset>/<Attack>/`.
You need to confirm that the target poisoned dataset has been generated in advance.

**Step 2: Calculate Unlearnable Distance (UD) of the poisoned model and poisoned datasets**: 

First, you need to select a model (e.g. Resnet-18) and do vanilla trainning with a clean dataset to obtain the clean model. Then, based on the SAL logs of the clean model and the poisoned model, calculate the Unlearnable Distance (UD).
```bash
python analysis/get_ud.py 
```

## License
This code is distributed under an [MIT LICENSE](LICENSE).  
Note that our code depends on other libraries, including PyTorch, and uses datasets that each have their own respective licenses that must also be followed.