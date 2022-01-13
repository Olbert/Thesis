from os.path import splitext
from os import listdir
import os
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import nibabel as nib
import cv2
from DomainVis.database_process.dataset_convert import preprocess_image
import h5py
import config
import skimage.transform as skTrans


class BasicDataset(Dataset):
	def __init__(self, imgs_dir, masks_dir, slices, size=(256, 256), mask_suffix='', preprocess=True):
		self.imgs_dir = imgs_dir
		self.masks_dir = masks_dir
		self.size = size
		self.mask_suffix = mask_suffix
		self.slices = slices

		self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
		            if not file.startswith('.')]
		logging.info(f'Creating dataset with {len(self.ids) * self.slices} examples')

	def __len__(self):
		return len(self.ids) * self.slices

	@classmethod
	def preprocess(cls, img, size):
		img = np.array(img)

		img = preprocess_image(img, size)
		if len(img.shape) == 2:
			img = np.expand_dims(img, axis=2)

		# HWC to CHW
		# img = img.transpose((2, 0, 1))
		if img.max() > 1:
			img = img / 255

		return img

	def __getitem__(self, i):
		idx = self.ids[np.int(i / self.slices)]
		# TODO change glob
		mask_file = glob(os.path.join(self.masks_dir, idx + self.mask_suffix + '.*'))
		img_file = glob(os.path.join(self.imgs_dir, idx + '.*'))

		assert len(mask_file) == 1, \
			f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
		assert len(img_file) == 1, \
			f'Either no image or multiple images found for the ID {idx}: {img_file}'

		# TODO: preprocessing before slice choosing?

		mask = nib.load(mask_file[0]).get_fdata()
		img = nib.load(img_file[0]).get_fdata()

		slice_num = (mask.shape[0] - self.slices) // 2 + np.int(i % self.slices)
		mask = mask[slice_num]
		img = img[slice_num]

		assert img.size == mask.size, \
			f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

		img = self.preprocess(img.reshape((-1,img.shape[0],img.shape[1])), self.size)
		mask = self.preprocess(mask.reshape((-1,mask.shape[0],mask.shape[1])), self.size)

		return {
			'image': torch.from_numpy(img).type(torch.FloatTensor),
			'mask': torch.from_numpy(mask).type(torch.FloatTensor)
		}


class NumpyDataset(BasicDataset):
	def __init__(self, imgs_dir, masks_dir, slices, size=(256, 256), mask_suffix=''):
		super().__init__(imgs_dir, masks_dir, slices, size, mask_suffix)
		self.imgs_dir = imgs_dir
		self.masks_dir = masks_dir
		self.size = size
		self.mask_suffix = mask_suffix
		self.slices = slices

		self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
		            if not file.startswith('.')]
		logging.info(f'Creating dataset with {len(self.ids) * self.slices} examples')

	def __getitem__(self, i):
		idx = self.ids[np.int(i / self.slices)]
		# TODO change glob
		mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
		img_file = glob(self.imgs_dir + idx + '.*')

		assert len(mask_file) == 1, \
			f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
		assert len(img_file) == 1, \
			f'Either no image or multiple images found for the ID {idx}: {img_file}'

		# TODO: preprocessing before slice choosing

		mask = np.load(mask_file[0])
		img = np.load(img_file[0])

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


class H5Dataset(Dataset):
	# TODO: What is class and what is instance variable?
	def __init__(self, name, imgs_dir, masks_dir, slices, size=(256, 256), mask_suffix=''):
		self.imgs_dir = imgs_dir
		self.masks_dir = masks_dir
		self.img_size = size
		# self.mask_suffix = mask_suffix
		self.slices = slices
		self.name = name

	def __getitem__(self, i):

		with h5py.File(self.imgs_dir, 'a') as hf:
			if self.name in hf.keys():
				image = hf[self.name][i]
				image = image.reshape(1, image.shape[0], image.shape[1])
				# image = image.reshape(image.shape[0], 1, image.shape[1])
				hf.close()
				if self.masks_dir is None:
					return {
						'image': torch.from_numpy(image).type(torch.FloatTensor),
						'mask': None
					}
				else:
					with h5py.File(self.masks_dir, 'a') as ms:
						true_mask = ms[self.name][i]
						true_mask = true_mask.reshape(1, true_mask.shape[0], true_mask.shape[1])
						# true_mask = true_mask.reshape(true_mask.shape[0], 1, true_mask.shape[1])
						ms.close()
						return {
							'image': torch.from_numpy(image).type(torch.FloatTensor),
							'mask': torch.from_numpy(true_mask).type(torch.FloatTensor)
						}

	def __len__(self):
		length = 0
		with h5py.File(self.imgs_dir, 'a') as hf:
			if self.name in hf.keys():
				length = hf[self.name].shape[0]
				hf.close()

		return length

	def keys(self):
		with h5py.File(self.imgs_dir, 'a') as hf:
			keys = hf.keys()
			hf.close()
		return keys

	def masks_exist(self):
		return self.masks_dir is not None \

	@staticmethod
	def combine(files, out_path):

		with h5py.File(out_path, mode='w') as h5fw:
			for h5name in files:
				h5fr = h5py.File(h5name, 'r')
				for dset in list(h5fr.keys()):
					arr_data = h5fr[dset][:]
					h5fw.create_dataset(dset, data=arr_data)
		return out_path

	@staticmethod
	def keys(imgs_dir):
		with h5py.File(imgs_dir, 'a') as hf:
			keys = list(hf.keys())
			hf.close()
		return keys

class BasicProcessedDataset(Dataset):
	def __init__(self, slices, size=(256, 256), imgs_dir=None, masks_dir=None, mask_suffix='', preprocess=True):
		self.imgs_dir = imgs_dir
		self.masks_dir = masks_dir
		self.size = size
		self.mask_suffix = mask_suffix
		self.slices = slices

		self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
		            if not file.startswith('.')]

		self.process(imgs_dir)
		if masks_dir is not None:
			self.process(masks_dir)

	def __len__(self):
		return len(self.ids) * self.slices

	def __getitem__(self, i):

		img_path = os.path.join(config.UPLOAD_DIRECTORY, config.PROCESSED_DIRECTORY, type + '.h5')

	def process(self, dir, type='data'):

		for filename in os.listdir(dir):
			img_file = os.path.join(dir, filename)
			if os.path.isfile(img_file):

				img = nib.load(img_file).get_fdata()

				img = np.array(img)
				if self.size is None:
					self.size = (img.shape[1], img.shape[2])

				img = img[(img.shape[0] - self.slices) // 2:(img.shape[0] + self.slices) // 2]

				img = preprocess_image(img, self.size)

				img = (255 * (img - np.min(img)) / np.ptp(img)).astype(int)

				name, img_num = filename.split('.')[0].split('_')

				path = os.path.join(config.UPLOAD_DIRECTORY, config.PROCESSED_DIRECTORY, type + '.h5')
				with h5py.File(path, 'a') as hf:

					if name in hf.keys():
						hf[name].resize((hf[name].shape[0] + img.shape[0]), axis=0)
						hf[name][-img.shape[0]:] = img
					else:
						hf.create_dataset(name, data=img, compression="gzip", chunks=True, maxshape=(None,))
					hf.close()
