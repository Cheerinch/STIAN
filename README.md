# STIAN: Subgraph Temporal Inconsistency Attention Network

Official PyTorch implementation of STIAN for unsafe behavior recognition based on RGB videos and skeleton sequences.

---

# Overview

STIAN (Subgraph Temporal Inconsistency Attention Network) is a dual-stream RGB-Skeleton fusion framework designed for unsafe behavior recognition.

The framework integrates:

* RGB appearance representation
* Skeleton motion representation
* Subgraph-level temporal inconsistency modeling
* Bias-guided Transformer attention

STIAN enhances temporal attention on critical unsafe motion frames by introducing temporal inconsistency as attention bias.

---

# Framework

## Main Components

* RGB Backbone (ResNet18)
* Skeleton GCN Encoder
* Subgraph Temporal Inconsistency Modeling
* Bias Transformer Encoder
* RGB-Skeleton Fusion

---

# Project Structure

```text
STIAN/
│
├── configs/
│   └── config.py
│
├── datasets/
│   └── fusion_dataset.py
│
├── models/
│   ├── fusion_model.py
│   ├── biased_transformer.py
│   └── graph.py
│
├── trainers/
│   └── trainer.py
│
├── utils/
│   ├── metrics.py
│   └── seed.py
│
├── checkpoints/
│
├── data/
│   └── NTU/
│       ├── videos/
│       └── skeletons/
│
├── main.py
├── requirements.txt
└── README.md
```

---

# Dataset

This project is evaluated on the NTU RGB+D dataset.

Datasets:

* NTU RGB+D 120

Official Website:

https://rose1.ntu.edu.sg/dataset/actionRecognition/

Please download the dataset from the official source.

---

# Dataset Organization

After downloading the dataset, organize it as:

```text
data/
└── NTU/
    ├── videos/
    │   ├── xxx.mp4
    │   └── ...
    │
    └── skeletons/
        ├── xxx.skeleton
        └── ...
```

---

# Environment

## Requirements

* Python 3.9+
* PyTorch 2.0+
* CUDA 11+
* torch-geometric
* timm
* OpenCV

---

# Installation

```bash
git clone https://github.com/anonymous/STIAN.git

cd STIAN

pip install -r requirements.txt
```

---

# Training

```bash
python main.py
```

---

# Model Checkpoints

Trained checkpoints will be saved to:

```text
checkpoints/
```

---


# Experimental Settings

* Backbone: ResNet18
* GCN Hidden Dim: 64
* Transformer Heads: 8
* Transformer Layers: 2
* Frames per Video: 32
* Optimizer: AdamW
* Scheduler: CosineAnnealingLR

---

# Acknowledgement

* NTU RGB+D Dataset
* PyTorch
* PyTorch Geometric
* timm

---

# License

This repository is released for anonymous academic review only.
