import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from DomainVis.unet.model import UNet2D
from DomainVis.unet.utils.data_vis import plot_img_and_mask
from DomainVis.database_process.dataset import BasicDataset



net = UNet2D(n_chans_in=1, n_chans_out=1)
model_path="E:\Thesis\conp-dataset\projects\calgary-campinas\CC359\Train2\checkpoints\CP_epoch200.pth"
net.load_state_dict(torch.load(model_path))

PATH = "unet_damri.pt"
torch.save(net, PATH)