# Individualized HRTF Prediction from Multiview 2D Ear Images

This repository contains the code and resources for predicting individualized Head-Related Transfer Functions (HRTFs) from multiview 2D images of the ear. This project was developed as part of the **Munich Tech Arena 2024** organized by **Huawei**, where it secured the **3rd prize**.

The repository includes scripts for training, inference, and evaluation, as well as utility functions and datasets.

This project was developped by Anas EZZAKRI & Houda GHALLAB.

![Alt text](images/pipeline.png)

---

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Repository Structure](#repository-structure)
4. [References](#references)

---

## Installation

To get started, install the necessary dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Next, follow the instructions provided in the official DepthAnything-V2 repository to set up the depth estimation model.

## Usage
To run the HRTF prediction model, run the file inference.py and execute the following script:

## Repository Structure
.
├── checkpoints/               # Pre-trained model checkpoints
│   ├── depth_anything_v2_vitl.pth
│   ├── HRTFNet.pth
│   └── mean_hrtf.pt
├── data/                      # Data processing scripts and exploratory analysis
│   ├── ear_extraction.py
│   └── EDA.ipynb
├── model/                     # Model definitions and utilities
│   ├── 3DHRTF.py
│   ├── HRTFNet_onefreq.py
│   ├── models.py
│   └── depth_anything_v2/     # DepthAnything-V2 model files
├── scripts/                   # Main scripts for training and inference
│   ├── inference.py
│   ├── test.py
│   └── training.ipynb
├── test_data/                 # Sample test images for inference
├── utils/                     # Utility functions and metrics
│   ├── metrics.py
│   ├── pointNet_utils.py
│   └── utils_d.py
├── .gitignore
├── README.md
├── requirements.txt
├── test_output.sofa
├── tree.txt
└── __init__.py

## References
