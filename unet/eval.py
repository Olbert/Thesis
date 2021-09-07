from os import listdir
from os.path import isfile, join
from os import walk
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import torch
import sklearn.manifold
import matplotlib.pyplot as plt
from torch.optim import Adam
from torchvision import models
from sklearn import preprocessing
from model import UNet2D
from misc_functions import preprocess_image, recreate_image, save_image
from PIL import Image
from utils.dataset import BasicDataset
from utils.dim_reduction_utils import TSNE, PCA, LLE, Isomap
from skimage.transform import resize
import argparse
import logging
import os
import sys
import platform

import numpy as np
import torch
import torch.nn as nn
import cv2
import scipy.misc
from torch import optim
from tqdm import tqdm

from model import UNet2D

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset, NumpyDataset
from torch.utils.data import DataLoader, random_split
from dim_reduction import get_mid_output

import platform

import torch
import torch.nn.functional as F
from tqdm import tqdm

from dice_loss import dice_coeff

import surface_distance


if platform.system()=='Windows': n_cpu= 0
def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_chans_out == 1 else torch.long
    # TODO: redefine length
    n_val = len(loader)  # the number of batch
    tot = 0

    # with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
    for batch in loader:
        imgs, true_masks = batch['image'], batch['mask']
        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=mask_type)

        with torch.no_grad():
            mask_pred = net(imgs)

        if net.n_chans_out > 1:
            tot += F.cross_entropy(mask_pred, true_masks).item()
        else:
            pred = torch.sigmoid(mask_pred)
            pred = (pred > 0.5).float()
            tot += dice_coeff(pred, true_masks).item()
        # pbar.update()

    net.train()
    return tot / n_val


if __name__ == '__main__':
    dir_img = 'E:\\Thesis\\conp-dataset\\projects\\calgary-campinas\\CC359\\Test3\\'

    """ Net setup """
    net = UNet2D(n_chans_in=1, n_chans_out=1)
    net.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device='cuda')
    model_path = "E:\Thesis\conp-dataset\projects\calgary-campinas\CC359\Train1\checkpoints\CP_epoch79.pth"
    net.load_state_dict(torch.load(model_path, map_location=device))
    slices = 50
    img_size = (128, 128)


    onlyfiles = [f for f in listdir(dir_img) if isfile(join(dir_img, f))]

    dirnames = np.array(walk(dir_img).__next__()[1])

    test_loaders = []


    # print("Dice score: ")
    # for name in dirnames:
    #     dataset = BasicDataset(os.path.join(dir_img, name), os.path.join(dir_mask, name), slices, img_size, '_ss')
    #     test_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    #     val_score = eval_net(net, test_loader, device)
    #
    #     print(str(name) + ': ' + str(val_score))


    print("Surface distance: ")
    for name in dirnames:
        # dataset = BasicDataset(os.path.join(dir_img, name), os.path.join(dir_mask, name), slices, img_size, '_ss')
        dataset = BasicDataset(os.path.join(dir_img, name, 'images/'), os.path.join(dir_img, name, 'masks/'), slices,
                               img_size, '_ss')

        test_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
        """Evaluation without the densecrf with the dice coefficient"""
        net.eval()
        mask_type = torch.float32 if net.n_chans_out == 1 else torch.long

        i = 0
        total = 0
        true_masks_all = []
        mask_pred_all = []
        for batch in test_loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.cpu().numpy()[0,0]
            true_masks_all.append(true_masks)

            with torch.no_grad():
                mask_pred = net(imgs)
                mask_pred = torch.sigmoid(mask_pred)
                mask_pred = mask_pred.cpu().numpy()[0,0]
                mask_pred = (mask_pred > 0.5)
                mask_pred_all.append(mask_pred)

        val_score = surface_distance.compute_surface_distances(np.array(mask_pred_all), np.array(true_masks_all, dtype=np.bool), [1,1,1])
        dice = surface_distance.compute_surface_dice_at_tolerance(val_score, 1)



        print(str(name) + ': ' + str(dice))


"""
Surface distance: 2mm philips_15 2D
ge_15: 0.7882200408190674
ge_3: 0.6297838360700679
philips_15: 0.87457095303606
philips_3: 0.2743725966714217
siemens_15: 0.7327510562328482
siemens_3: 0.8081202860991571
"""


"""
Surface distance: 1mm philips_15 2D
ge_15: 0.4241895847958042
ge_3: 0.3949352637951144
philips_15: 0.35987503172707885
philips_3: 0.062380443667114455
siemens_15: 0.37092389382350954
siemens_3: 0.45467486734707013
"""

""" 1mm philips_15 3D
Surface distance: 
ge_15: 0.6877610776228444
ge_3: 0.7427685547441867
philips_15: 0.7324438841707778
philips_3: 0.24514733242980716
siemens_15: 0.6734573380780277
siemens_3: 0.717354212164526
"""


""" 1mm philips_15 3D multiple
ge_15: 0.6291184916710084
ge_3: 0.9012111533221929
philips_15_test: 0.7968217265060121
philips_15_train: 0.8302636824647416
philips_3: 0.22820770862443862
siemens_15: 0.8306324651525312
siemens_3: 0.8582756925217326
"""