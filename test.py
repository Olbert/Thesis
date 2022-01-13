import argparse
import logging
import os
import sys
import platform
import h5py
import numpy as np
import torch
import torch.nn as nn
import cv2
import scipy.misc
from torch import optim
from tqdm import tqdm

from DomainVis.unet.eval import eval_net
from DomainVis.unet.model import UNet2D


from torch.utils.tensorboard import SummaryWriter
from DomainVis.database_process.dataset import BasicDataset, NumpyDataset, H5Dataset
from DomainVis.database_process.dataset_convert import convert_to_h5
from torch.utils.data import DataLoader, random_split



dir_img = "E:\Thesis\conp-dataset\projects\calgary-campinas\CC359\Ready\images"
dir_mask = "E:\Thesis\conp-dataset\projects\calgary-campinas\CC359\Ready\masks"
dir_checkpoint = 'E:\\Thesis\\conp-dataset\\projects\\calgary-campinas\\CC359\\Train3\\checkpoints/'
if platform.system() == 'Windows': n_cpu= 0

dir_img_out= "E:\\Thesis\\gdrive"
dir_mask_out= "E:\\Thesis\\gdrive"

total = {'philips_15':0,
         'philips_3':0,
         'siemens_15':0,
         'siemens_3':0,
         'ge_15':0,
         'ge_3':0
         }

i=0

with h5py.File('E:\\Thesis\\gdrive\\train\\ge_3\\data.h5', 'a') as hf:
    pass

for path, currentDirectory, files in os.walk(dir_img):
    for file in tqdm(files):

            name = file.split('.')[0].split('_')[1]+"_"+file.split('.')[0].split('_')[2]

            if total[name]<55:
                out_directory = os.path.join(dir_img_out, 'train', name)

                if not os.path.exists(out_directory):
                    os.makedirs(out_directory)
                convert_to_h5(dir_img, out_directory, 'data', name=file)
                convert_to_h5(dir_mask, out_directory, 'masks', name=file.split('.')[0]+"_ss.nii")
            elif total[name]<59:
                out_directory = os.path.join(dir_img_out, 'test', name)

                if not os.path.exists(out_directory):
                    os.makedirs(out_directory)
                convert_to_h5(dir_img, out_directory, 'data', name=file)
                convert_to_h5(dir_mask, out_directory, 'masks', name=file.split('.')[0]+"_ss.nii")

            total[name] += 1

            i+=1
