from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import nibabel as nib
import cv2
class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, slices, size=(256, 256), mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.size = size
        self.mask_suffix = mask_suffix
        self.slices = slices

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)*self.slices} examples')

    def __len__(self):
        return len(self.ids)*self.slices

    @classmethod
    def preprocess(cls, pil_img, size):
        pil_img = np.array(pil_img)
        assert size[0] > 0 and size[1] > 0, 'Scale is too small'
        # TODO Faster solution
        img_nd = pil_img.copy()
        img_nd = cv2.resize(img_nd, size)

        img_nd = np.array(img_nd)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[np.int(i/self.slices)]
        # TODO change glob
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'

        # TODO: preprocessing before slice choosing

        mask = nib.load(mask_file[0]).get_fdata()
        img = nib.load(img_file[0]).get_fdata()

        slice_num = (mask.shape[0] - self.slices) // 2 + np.int(i % self.slices)
        mask = mask[slice_num]
        img = img[slice_num]

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.size)
        mask = self.preprocess(mask, self.size)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, size=(256,256)):
        super().__init__(imgs_dir, masks_dir, size, mask_suffix='_mask')