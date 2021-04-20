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

from eval import eval_net
from model import UNet2D

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset, NumpyDataset
from torch.utils.data import DataLoader, random_split

dir_img = 'E:\\Thesis\\conp-dataset\\projects\\calgary-campinas\\CC359\\Test\\images/'
dir_mask = 'E:\\Thesis\\conp-dataset\\projects\\calgary-campinas\\CC359\\Test\\masks/'
dir_checkpoint = 'E:\\Thesis\\conp-dataset\\projects\\calgary-campinas\\CC359\\Test\\checkpoints/'
if platform.system() == 'Windows': n_cpu= 0

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 80 epochs"""
    lr = args.lr * (0.1 ** (epoch // 80))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=1):
    # TODO: change place
    slices = 60
    dataset = BasicDataset(dir_img, dir_mask, slices, img_scale,  '_ss')



    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    criterion, n_filters, batch_size, batches_per_epoch, n_epochs, lr_init, optimizer = net.default()


    optimizer = optimizer(
        net.parameters(),
        lr=lr_init,
        momentum=0.9,
        nesterov=True,
        weight_decay=0
    )
    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_chans_out > 1 else 'max', patience=2)
    #
    # if net.n_chans_out > 1:
    #     criterion = nn.CrossEntropyLoss()
    # else:
    #     criterion = nn.BCEWithLogitsLoss()


    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.n_chans_out, \
                    f'Network has been defined with {net.n_chans_out} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_chans_out == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)

                adjust_learning_rate(optimizer, epoch)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (max(n_train // (10 * batch_size), 300)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    if (len(val_loader)!=0):
                        val_score = eval_net(net, val_loader, device)
                        # scheduler.step(val_score)

                        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                        if net.n_chans_out > 1:
                            logging.info('Validation cross entropy: {}'.format(val_score))
                            writer.add_scalar('Loss/test', val_score, global_step)
                        else:
                            logging.info('Validation Dice Coeff: {}'.format(val_score))
                            writer.add_scalar('Dice/test', val_score, global_step)

                        writer.add_images('images', imgs, global_step)
                        if net.n_chans_out == 1:
                            writer.add_images('masks/true', true_masks, global_step)
                            writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=200,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=32,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--size', dest='size', type=float, default=(128,128),
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_chans_out is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_chans_out=1
    #   - For 2 classes, use n_chans_out=1
    #   - For N > 2 classes, use n_chans_out=N
    net = UNet2D(n_chans_in=1, n_chans_out=1)
    # logging.info(f'Network:\n'
    #              f'\t{net.n_chans_in} input channels\n'
    #              f'\t{net.n_chans_out} output channels (classes)\n'
    #              f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # TODO: Research this
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.size,
                  val_percent=np.float(args.val) / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
