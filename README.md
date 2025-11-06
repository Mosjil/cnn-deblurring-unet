This project is part of an assignment for the Computational imaging course at POSTECH (South Korea)

# CNN Deblurring U-Net

This repository contains an implementation of a Convolutional Neural Network (CNN) using the U-Net architecture for image deblurring.

## Overview

The aim of this project is to restore sharp images from blurry inputs using deep learning techniques. Blurry images are automatically generated using **Pado**, which python library used for computational imaging. Here it simulates various blur types. Users only need to import their sharp images; the process of generating corresponding blurry images is handled by the codebase.

## Features

- U-Net architecture for image-to-image deblurring
- Automatic generation of blurry images via Pado
- Easy dataset preparation: simply import your sharp images
- Training and evaluation scripts for custom datasets
- Visualization tools for comparing sharp, blurry, and deblurred images

## Installation

Clone the repository:

```bash
git clone https://github.com/Mosjil/cnn-deblurring-unet.git
cd cnn-deblurring-unet
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Dataset Preparation

1. Put your sharp images in a dicrectory, be sure to update the config.py.
2. The code will automatically use Pado to create matching blurry image pairs.

### Training

Train the model by running:

```bash
python train.py --epochs 100 --batch-size 8 --learning-rate 1e-4
```

You can modify the arguments to suit your needs.

### Evaluation

To evaluate the trained model:

```bash
python evaluate.py 
```
