# Blurry-Edges: Photon-Limited Depth Estimation from Defocused Boundaries

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/) [![Accepted to CVPR 2025](https://img.shields.io/badge/Accepted-CVPR%202025-367DBD?style=flat&logo=ieee)](https://cvpr.thecvf.com/Conferences/2025) [![Project Webpage](https://img.shields.io/badge/Project%20Webpage-Blurry--Edges-green)](https://blurry-edges.qiguo.org/) [![arXiv](https://img.shields.io/badge/arXiv-2503.23606-red)](https://arxiv.org/abs/2503.23606) [![PyTorch](https://img.shields.io/badge/Implemented%20with-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/) [![GitHub stars](https://img.shields.io/github/stars/guo-research-group/Blurry-Edges?style=social)](https://github.com/guo-research-group/Blurry-Edges/stargazers)

*[Wei Xu](https://www.weixu.xyz/), [Charles James Wagner](https://www.linkedin.com/in/charlie-wagner-887284221/), [Junjie Luo](https://luo-jun-jie.github.io/), and [Qi Guo](https://qiguo.org)*

Elmore Family School of Electrical and Computer Engineering, Purdue University

Contact: weixu@purdue.edu

<a href="https://blurry-edges.qiguo.org/" title="Blurry-Edges project webpage">**Project webpage**</a> | <a href="https://arxiv.org/abs/2503.23606" title="Blurry-Edges arXiv">**arXiv**</a> | <a href="https://blurry-edges.qiguo.org/data" title="Blurry-Edges dataset">**Dataset (including specifications)**</a> | <a href="https://purdue0-my.sharepoint.com/:f:/g/personal/xu1639_purdue_edu/EqD_pFLP0d9MvTJVwXrq-8kBwrkd4IioWmzNKOiWR2DEwg" title="Blurry-Edges pretrained weights">**Pre-trained weights**</a>

**Content**

- [0 Introduction](#0-introduction)
- [1 Usage](#1-usage)
  * [1.1 Configure environment](#11-configure-environment)
  * [1.2 Training](#12-training)
  * [1.3 Testing and evaluation](#13-testing-and-evaluation)
  * [1.4 Regenerate testing set](#14-regenerate-testing-set)
- [2 Reference](#2-reference)
- [3 Acknowledgement](#3-acknowledgement)

## 0 Introduction

Blurry-Edges is a new image patch representation that explicitly stores and visualizes a rich set of low-level patch information, including boundaries, color, and smoothness. We develop a deep neural network architecture that predicts the Blurry-Edges representation from a pair of differently defocused images, from which depth can be analytically calculated using a novel DfD relation we derive.

![Overview](/pic/teaser.png "Overview")

## 1 Usage

### 1.1 Configure environment

Create a conda environment, please follow the prompts below. 
```
git clone https://github.com/guo-research-group/Blurry-Edges.git
cd Blurry-Edges
conda create -n blurry_edges python=3.9
conda activate blurry_edges
pip3 install torch torchvision torchaudio
pip3 install -r requirements.txt
```

The training, validation, testing, and real captured data can be downloaded through <a href="https://blurry-edges.qiguo.org/data" title="Blurry-Edges dataset">this page</a>. In the meantime, users can also generate own training, validation, and testing sets by following the steps outlined below.

The pre-trained weights can be downloaded through <a href="https://purdue0-my.sharepoint.com/:f:/g/personal/xu1639_purdue_edu/EqD_pFLP0d9MvTJVwXrq-8kBwrkd4IioWmzNKOiWR2DEwg" title="Blurry-Edges dataset">OneDrive</a>.

### 1.2 Training

Prepare the training and validation set containing only basic shapes.

    python train_val_data_generator.py

Train local stage.

    python local_training.py

Pre-calculate parameters to accelerate global stage training.

    python global_data_pre_cal.py

Train global stage.

    python global_training.py

### 1.3 Testing and evaluation

Test with synthetic data containing realistic textures.

    python blurry_edges_test.py

Test with synthetic data containing realistic textures and output dense depth maps by assigning depth values to wedges.

    python blurry_edges_test.py --densify 'w'

Test with synthetic data containing realistic textures and output dense depth maps by U-Net post-prosseccing.

    python blurry_edges_test.py --densify 'pp'

Test with large-size synthetic data using blocks.

    python blurry_edges_test_big.py

### 1.4 Regenerate testing set

Users can generate the testing set (both regular and large size) by following steps. First, download [The Painting dataset](https://www.robots.ox.ac.uk/~vgg/data/paintings/) to `./data/Painting` and [MS COCO dataset](https://cocodataset.org/) (for our experiments, we only use `2017 Val` images and `instances_val2017` annotations) to `./data/MS_COCO_annotations`. To generate the testing set with regular image size, run the following command:

    python test_data_generator.py

To generate the set with large image size, set `BIG = True`. We recommend using an image size of $147 + 4x$, where $x = 0, 1, 2, \cdots$. Users should also update `big_img_size` and `n_margin_patch` arguments accordingly. 

## 2 Reference

```
@inproceedings{xu2025blurry,
        title={Blurry-Edges: Photon-Limited Depth Estimation from Defocused Boundaries},
        author={Xu, Wei and Wagner, Charles James and Luo, Junjie and Guo, Qi},
        booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
        pages={432--441},
        year={2025}
      }
```

## 3 Acknowledgement

We thank [Hyunseung (Seth) Eom](https://www.linkedin.com/in/eomh/), [Lucas Nguyen](https://www.linkedin.com/in/lucasnguyen1207/), and [Soumil Verma](https://www.linkedin.com/in/soumilverma/) for their contributions to the analysis and improvement of the code.