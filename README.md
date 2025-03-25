# Blurry-Edges: Photon-Limited Depth Estimation from Defocused Boundaries

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/) ![Accepted to CVPR 2025](https://img.shields.io/badge/Accepted-CVPR%202025-green?style=flat&logo=ieee)

*[Wei Xu](https://www.weixu.xyz/), [Charles James Wagner](https://www.linkedin.com/in/charlie-wagner-887284221/), [Junjie Luo](https://luo-jun-jie.github.io/), and [Qi Guo](https://qiguo.org)*

Elmore Family School of Electrical and Computer Engineering, Purdue University

Contact: xu1639@purdue.edu

<a href="https://blurry-edges.qiguo.org/" title="Blurry-Edges project webpage">**Project webpage**</a> | <a href="https://drive.google.com/drive/folders/1kteggKmnYCLIYnCmNTyzCoKp3NM7WZ6V?usp=sharing" title="Blurry-Edges testing dataset">**Testing dataset**</a>

**Content**

- [0 Introduction](#0-introduction)
- [1 Usage](#1-usage)
  * [1.1 Configure environment](#11-configure-environment)
  * [1.2 Training](#12-training)
  * [1.3 Testing and evaluation](#13-testing-and-evaluation)
- [2 Citation](#3-citation)

## 0 Introduction

Blurry-Edges is a new image patch representation, Blurry-Edges, that explicitly stores and visualizes a rich set of low-level patch information, including boundaries, color, and smoothness. We develop a deep neural network architecture that predicts the Blurry-Edges representation from a pair of differently defocused images, from which depth can be analytically calculated using a novel DfD relation we derive.

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

We are seeking a reliable server to host our training and validation sets. *We will update this page once it's available. Stay tuned!* In the meantime, users can still generate these sets by following the steps outlined [below](#12-training). The testing sets are available for download via <a href="https://drive.google.com/drive/folders/1kteggKmnYCLIYnCmNTyzCoKp3NM7WZ6V?usp=sharing" title="fieldofjunctions">Google Drive</a>. The default paths of regular and big testing sets are `./data/data_test` and `./data/data_test_big`, respectively. 

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

Test with large-size synthetic data using blocks.

    python blurry_edges_test_big.py

## 2 Citation

```
@InProceedings{xu2025blurryedges,
        author    = {Xu, Wei and Wagner, Charles James and Luo, Junjie and Guo, Qi},
        title     = {Blurry-Edges: Photon-Limited Depth Estimation from Defocused Boundaries},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        month     = {June},
        year      = {2025}
    }
```
