import os
import numpy as np

import torch

import sklearn.manifold
import matplotlib.pyplot as plt

from sklearn import preprocessing

from PIL import Image

import sklearn.manifold

from torch.utils.tensorboard import SummaryWriter
from DomainVis.database_process.dataset import BasicDataset, NumpyDataset, BasicProcessedDataset, H5Dataset
from torch.utils.data import DataLoader, random_split


import config

np.random.seed(seed=0)
scaler = preprocessing.MinMaxScaler()
torch.random.manual_seed(seed=0)
activation = {}
weight = {}


def get_activation(name):
	def hook(model, input, output):
		activation[name] = output.detach()

	return hook


def get_filter(name):
	def hook(model, input, output):
		weight[name] = output.detach()

	return hook


def remove_far(input):  # unused
	mean = np.absolute(np.mean(input, axis=0)).sum()
	sd = np.absolute(np.std(input, axis=0)).sum()

	final_list = [x for x in input if (np.absolute(x).sum() > mean - 2 * sd)]
	final_list = [x for x in final_list if (np.absolute(x).sum() < mean + 2 * sd)]
	return np.array((final_list))


def get_mid_output(net, img, layer='init_path', device='cuda', sigmoid=True):
	getattr(net, layer).register_forward_hook(get_activation(layer))  # get the layer by it's name
	net.eval()
	img = img.to(device=device, dtype=torch.float32)

	with torch.no_grad():
		raw_out = net(img)
		if sigmoid:
			output = torch.sigmoid(raw_out).cpu().detach().numpy()[0, 0]
		else:
			output = raw_out.cpu().detach().numpy()[0, 0]
		output_mid = activation[layer].cpu().detach().numpy()[0]

	return output_mid, output


def scatter_output(output_mid, col):
	output_mid = output_mid.reshape(16, -1)
	output_2d = sklearn.manifold.TSNE(n_components=2, perplexity=5).fit_transform(output_mid).astype(np.float)
	# output_2d = remove_far(output_2d)
	plt.scatter(x=output_2d[:, 0], y=output_2d[:, 1], c=col)


# ax.scatter(*zip(*output_2d), c=col)


def pixel_to_coordinate(output_mid, threshold=0.5):
	coord = []
	for output in output_mid:
		for i in range(output.shape[0]):
			for k in range(output.shape[1]):
				if output[i, k] > threshold:
					coord.append([i, k])
		return np.array(coord)


def threshold_transform(output_mid, threshold=0):  # TODO: more efficient way
	for output in output_mid:
		for i in range(output.shape[0]):
			for k in range(output.shape[1]):
				if output[i, k] > threshold:
					output[i, k] = 1
				else:
					output[i, k] = 0
	return output_mid


def coord_scatter(output, axis, perp, col):
	scaler = preprocessing.MinMaxScaler()
	output = pixel_to_coordinate(output)
	output = scaler.fit_transform(output)
	output_2d = sklearn.manifold.TSNE(n_components=2, perplexity=perp).fit_transform(output).astype(
		np.float)

	output_2d = scaler.fit_transform(output_2d)

	axis.scatter(x=output_2d[:, 0], y=output_2d[:, 1], c=col, s=20)
	return output_2d


def show_images(images, folder="E:\\Thesis\\unet\\data\\images\\"):
	k = 0
	for image in images:
		for i in range(0, image.shape[0]):
			im = Image.fromarray(image[i].astype(np.uint8))
			im.save(folder + "image" + str(k) + '_filter' + str(i) + ".jpg")
			im.close()
			k += 1


def crop_mask(image_mid, mask):  # TODO: Find more efficient way
	if image_mid.shape[1] == mask.shape[0]:
		for i in range(0, image_mid.shape[0]):
			image_mid[i] = image_mid[i] * (mask > 0.5)

	return image_mid


def resize_image(image, size):
	if len(image.shape) == 3:
		new_image = np.zeros((image.shape[0], size[0], size[1]))
		for filter_pos in range(0, image.shape[0]):
			new_image[filter_pos] = nearest_neighbor_scaling(image[filter_pos], size)
	elif len(image.shape) == 2:
		new_image = nearest_neighbor_scaling(image, size)
	else:
		new_image = None
	return new_image


def nearest_neighbor_scaling(source, size):
	target = np.zeros(size)
	for x in range(0, size[0]):
		for y in range(0, size[1]):
			srcX = int(round(float(x) / float(size[0]) * float(source.shape[0])))
			srcY = int(round(float(y) / float(size[1]) * float(source.shape[1])))
			srcX = min(srcX, source.shape[0] - 1)
			srcY = min(srcY, source.shape[1] - 1)
			target[x, y] = source[srcX, srcY]

	return target


def get_data(domains, volume_num=1, sample_num=1, data_path="", mask_path="", ids=None):
	SIZE = (128, 128)
	""" Images """
	# folders = 'E:\\Thesis\\conp-dataset\\projects\\calgary-campinas\\CC359\\Test2\\'

	test_loaders = []
	if data_path == "":
		data_path = os.path.join(config.UPLOAD_DIRECTORY, config.PROCESSED_DIRECTORY, 'data.h5')
		mask_path = os.path.join(config.UPLOAD_DIRECTORY, config.PROCESSED_DIRECTORY, 'label.h5')

	masks_exist = os.path.exists(mask_path)  # len(next(walk(path), (None, None, []))[2])

	for name in domains:
		# TODO: Check image  size. Make constant
		if masks_exist:
			dataset = H5Dataset(name,
			                    data_path,
			                    mask_path,
			                    config.SLICES,
			                    SIZE)
		else:
			dataset = H5Dataset(name,
			                    data_path,
			                    None,
			                    config.SLICES,
			                    SIZE)
		# TODO: custom volumes and slices
		volumes = np.random.choice(range(len(dataset) // config.SLICES), volume_num, replace=False)
		subset = []
		for k in range(0, volumes.shape[0]):
			subset.append(
				np.random.choice(range(config.SLICES * volumes[k], config.SLICES * (volumes[k] + 1)), sample_num,
				                 replace=False))

		dataset_sub = torch.utils.data.Subset(dataset, np.array(subset).flatten())
		test_loaders.append(DataLoader(dataset_sub, batch_size=1, shuffle=True, num_workers=2, pin_memory=True))
	test_loaders = np.array(test_loaders)

	return test_loaders


def c_to_n(coord, arr_size):
	return coord[0] * arr_size + coord[1]


def n_to_c(num, arr_size):
	x = num // arr_size
	y = num - x * arr_size
	return x, y


def get_images_by_id(net, layer_name, id, path):
	test_loaders = get_data(path, 4)

	image_mids_temp = []
	image_mids = []

	true_masks = []
	names = []

	test_loader = test_loaders[id]

	image_mids_temp = []
	true_masks = []

	names.append(test_loader[1])
	test_loader = test_loader[0]
	batch_id = 0  # TODO: Find better solution for iteration
	for batch in test_loader:
		image, true_mask = batch['image'], batch['mask']
		image = image.reshape(image.shape[0], 1, image.shape[1], image.shape[2])
		true_mask = true_mask.reshape(true_mask.shape[0], 1, true_mask.shape[1], true_mask.shape[2])
		true_mask = torch.sigmoid(true_mask)
		true_mask = true_mask.cpu().numpy()[0, 0]
		true_masks.append(true_mask)

		image_mid, net_output = get_mid_output(net, image, layer_name)

		if batch_id == 0:
			batch_id += 1
			continue
		im = Image.fromarray((true_mask * 255).astype(np.uint8))
		im.save("mask_" + str(names[-1]) + "_" + str(batch_id) + ".jpg")

		im = Image.fromarray((net_output * 255).astype(np.uint8))
		im.save("output_" + str(names[-1]) + "_" + str(batch_id) + ".jpg")

		im = Image.fromarray(((image.cpu().numpy()[0, 0]) * 255).astype(np.uint8))
		im.save("original_" + str(names[-1]) + "_" + str(batch_id) + ".jpg")
		batch_id += 1

		break

		batch_id += 1
	image_mids_temp = np.array(image_mids_temp)

	image_mids.append(image_mids_temp)

	return image_mids, names
