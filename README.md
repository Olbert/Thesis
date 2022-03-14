# Visualizing the effects of domain shift on CNN-based image segmentation

### [Google Slides](https://docs.google.com/presentation/d/1YMiigPILl7YXAIzMO3UTdJRwpRNKXH82aXwTqcPlezs/edit?usp=sharing) 
<img title="" src="docs/figs/thanks.gif" alt="">

 In this work, we are trying to apply dimensionality reduction techniques to make better visualizations of inner representations of pretrained U-Net models. 
 The final goal of this work is to create a tool that can help to explore the domain shift problem in a given U-Net model.
 
 ![Alt text](images/page1_3_activ.png?raw=true "Title")
 
 
## Table of contents

-----

* [Installation](#requirements-and-installation)
* [Dataset](#dataset)
* [Training](#train-a-new-model)
* [Visualisation](#visualise-a-model)
* [License](#license)
* [Citation](#citation)

------

## Requirements and Installation

This code is implemented in PyTorch 

The code has been tested on the following system:

* Python 3.7
* PyTorch 1.8.0
* Nvidia GPU (RTX 1060) CUDA 10.1

Only training and rendering on GPUs are supported.

To install better use conda environment: first clone this repo and install conda:

```bash
wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
sh Anaconda3-2020.07-Linux-x86_64.sh -b -p conda/ -f
source conda/bin/activate
```

Then install conda dependencies:

```bash
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.1 -y -c pytorch
```

Then install pip dependencies:

```bash
pip install -r requirements.txt
```

Then install this project module in editable mode:

```bash
pip install --editable ./
```

## Dataset

A publicly available training subset of the Calgary-Campinas (CC-359) dataset was used for training and all experiments.
It can be downloaded here: [https://sites.google.com/view/calgary-campinas-dataset/download]
### Prepare your own dataset

## Train a new model

The `DomainVis/unet/train.py` is the script for compiling the command to run training. 


## Visualise a model

The `DomainVis/dash_interface/app.py` is the script for launching local flask server. 


## License

This work is MIT-licensed.
The license applies to the pre-trained models as well.

## Citation

Please cite as 

```bibtex
@mastersthesis{domain_shift,
  author       = {Albert Gubaidullin}, 
  title        = {Visualizing the effects of domain shift on CNN-based image segmentation},
  school       = {University of Bonn},
  year         = 2022
}
```
