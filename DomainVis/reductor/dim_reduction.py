import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import torch
import sklearn.manifold
import matplotlib.pyplot as plt
import pandas as pd
from torch.optim import Adam
from torchvision import models
from sklearn import preprocessing
from DomainVis.unet.model import UNet2D
from DomainVis.unet.misc_functions import preprocess_image, recreate_image, save_image
from PIL import Image
from DomainVis.database_process.dataset import BasicDataset
from DomainVis.reductor.dim_reduction_utils import TSNE, PCA, LLE, Isomap, PCA_cuda
from skimage.transform import resize

from torch.utils.tensorboard import SummaryWriter
from DomainVis.database_process.dataset import BasicDataset, NumpyDataset
from torch.utils.data import DataLoader, random_split

from os import listdir
from os.path import isfile, join
from os import walk
import json

np.random.seed(seed=0)

torch.random.manual_seed(seed=0)
activation = {}


def get_activation(name):
	def hook(model, input, output):
		activation[name] = output.detach()

	return hook


def remove_far(input):  # unused
	mean = np.absolute(np.mean(input, axis=0)).sum()
	sd = np.absolute(np.std(input, axis=0)).sum()

	final_list = [x for x in input if (np.absolute(x).sum() > mean - 2 * sd)]
	final_list = [x for x in final_list if (np.absolute(x).sum() < mean + 2 * sd)]
	return np.array((final_list))


def get_mid_output(net, img, layer='init_path', device='cuda'):
	# img = Image.open(os.path.join(os.getcwd(), 'data', 'input_s.jpg'))
	getattr(net, layer).register_forward_hook(get_activation(layer))  # get the layer by it's name
	net.eval()
	# full_img = np.array(img)
	# img2 = torch.from_numpy(BasicDataset.preprocess(full_img, (128, 128)))
	# img2 = img2.unsqueeze(0)
	img = img.to(device=device, dtype=torch.float32)

	with torch.no_grad():
		raw_out = net(img)
		output = torch.sigmoid(raw_out).cpu().detach().numpy()[0, 0]
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

	# plt.imshow(image_mid[i], cmap='gray', vmin=0, vmax=1)
	# plt.show()
	# plt.imshow(mask, cmap='gray', vmin=0, vmax=1)
	# plt.show()
	# plt.imshow(image_mid[i] * (mask > 0.5), vmin=0, vmax=1)
	# plt.show()
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


def get_data(volume_num, sample_num):
	# image_sources = np.array(['input_p_1.jpg', 'input_s.jpg', 'input_p_2.jpg'])
	# image_legends = np.array(['Source domain', 'Target domain', 'Source domain2'])
	SIZE = (128, 128)
	SLICES = 40
	""" Images """
	folders = 'E:\\Thesis\\conp-dataset\\projects\\calgary-campinas\\CC359\\Test2\\'

	foldernames = np.array(walk(folders).__next__()[1])

	test_loaders = np.empty((foldernames.shape[0]),dtype=DataLoader)
	for i in range(0, foldernames.shape[0]):
		# TODO: Check image  size. Make constant
		dataset = BasicDataset(os.path.join(folders, foldernames[i], 'images/'),
		                       os.path.join(folders, foldernames[i], 'masks/'), SLICES,
		                       SIZE, '_ss')
		volumes = np.random.choice(range(len(dataset.ids)), volume_num, replace=False)
		subset = []
		for k in range(0, volumes.shape[0]):
			subset.append(np.random.choice(range(SLICES * volumes[k], SLICES * (volumes[k]+1)), sample_num, replace=False))

		dataset_sub = torch.utils.data.Subset(dataset, np.array(subset).flatten())
		test_loaders[i] = (DataLoader(dataset_sub, batch_size=1, shuffle=True, num_workers=4, pin_memory=True), foldernames[i])
	test_loaders = np.array(test_loaders)

	return test_loaders


def get_map(mask, pred, coeff=0.5):  # TODO: check for errors
	"""
		0: True Positive
		1: True Negative
		2: False Positive
		3: False Negative
	"""

	eval_map = np.zeros(mask.shape)
	for i in range(0, mask.shape[0]):
		for k in range(0, mask.shape[1]):
			if mask[i, k] > 0.5:
				if pred[i, k] > coeff:
					eval_map[i, k] = 0
				else:
					eval_map[i, k] = 2
			if mask[i, k] == 0.5:
				if pred[i, k] < coeff:
					eval_map[i, k] = 1
				else:
					eval_map[i, k] = 3

	# plt.imshow(mask, cmap='gray', vmin=0, vmax=1)
	# plt.show()
	# plt.imshow(pred, cmap='gray', vmin=0, vmax=1)
	# plt.show()
	# plt.imshow(eval_map, vmin=2, vmax=3)
	# plt.show()

	return eval_map


def c_to_n(coord, arr_size):
	return coord[0] * arr_size + coord[1]


def n_to_c(num, arr_size):
	x = num // arr_size
	y = num - x * arr_size
	return x, y


def get_images_by_id(net, layer_name, id):
	test_loaders = get_data(4)

	image_mids_temp = []
	image_mids = []

	true_masks = []
	names = []

	test_loader = test_loaders[id[0]]

	image_mids_temp = []
	true_masks = []

	names.append(test_loader[1])
	test_loader = test_loader[0]
	batch_id = 0  # TODO: Find better solution for iteration
	for batch in test_loader:
		image, true_mask = batch['image'], batch['mask']
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


class BasePresenter():


	def get_database(self, net, test_loaders, layer_name, threshold, img_size, mask_cut, upsampling, plot_mode):

		for test_loader in test_loaders:
			image_mids_temp = []
			eval_maps_temp = []
			true_masks = []
			input_temp = []
			self.names.append(test_loader[1])
			test_loader = test_loader[0]
			batch_id = 0  # TODO: Find better solution for iteration
			for batch in test_loader:
				image, true_mask = batch['image'], batch['mask']
				true_mask = torch.sigmoid(true_mask)
				true_mask = true_mask.cpu().numpy()[0, 0]
				true_masks.append(true_mask)

				image_mid, net_output = get_mid_output(net, image, layer_name)

				if threshold is not None:
					image_mid = threshold_transform(image_mid, threshold)
				"""Evaluation map preparation"""

				if image_mid.shape[1] != true_mask.shape[0]:
					true_mask = resize_image(true_mask, (image_mid.shape[1], image_mid.shape[2]))

				if image_mid.shape[1] != net_output.shape[0]:
					net_output = resize_image(net_output, (image_mid.shape[1], image_mid.shape[2]))

				eval_map = get_map(true_mask, net_output)

				if img_size is not None:
					if image_mid.shape[1] > img_size[0] | upsampling:
						# if upsampling mode  or initial image is bigger than requested
						image_mid = resize_image(image_mid, img_size)
						eval_map = resize_image(eval_map, img_size)
						true_mask = resize_image(true_mask, img_size)

				if mask_cut == 'true':
					image_mid = crop_mask(image_mid, true_mask)
				elif mask_cut == 'predict':
					image_mid = crop_mask(image_mid, net_output)

				"""Plot mode"""
				if batch_id % plot_mode == 0:
					image_mids_temp.append(image_mid / plot_mode)
				else:
					image_mids_temp[-1] += image_mid / plot_mode

				eval_maps_temp.append(eval_map)  # no plot_mode considered
				input_temp.append(image.numpy())
				batch_id += 1

			image_mids_temp = np.array(image_mids_temp)
			eval_maps_temp = np.array(eval_maps_temp)
			input_temp = np.array(input_temp)

			self.image_mids.append(image_mids_temp)
			self.eval_maps.append(eval_maps_temp)
			self.input.append(input_temp)

		self.image_mids = np.array(self.image_mids)
		self.eval_maps = np.array(self.eval_maps)
		self.input = np.array(self.input)
		self.names = np.array(self.names)

class Reductor(BasePresenter):
	def __init__(self,
	             model_path="E:\Thesis\conp-dataset\projects\calgary-campinas\CC359\Train2\checkpoints\CP_epoch200.pth"):

		"""Parameters Init"""
		self.names = []

		self.image_mids = []
		self.eval_maps = []
		self.input = []
		self.output = []
		self.input_img_size = (128, 128)
		self.activ_img_size = (8, 8)
		self.coord = np.zeros(self.activ_img_size)

		""" Net setup """
		self.net = UNet2D(n_chans_in=1, n_chans_out=1)
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.net.to(device=device)
		self.net.load_state_dict(torch.load(model_path, map_location=device))


	def reduction(self, algos, modes, layer_name, threshold, img_size, volume_num, sample_num, mask_cut, perp=None, n_iter=500,
	              save=False):

		upsampling = False

		self.activ_img_size = img_size
		img_types = ['TP', 'TN', 'FP', 'FN']

		"""Plot mode structure """  # number of images to stack on each other (Doesn't work)
		plot_mode = 1
		desired_points = 200000  # Works not everywhere
		test_loaders = get_data(volume_num, sample_num)

		self.get_database(self.net, test_loaders, layer_name, threshold, img_size, mask_cut, upsampling, plot_mode)

		""" Dimensionality reduction """
		for algo in algos:
			for mode in modes:
				if algo == 'tsne':
					orig_shape = self.image_mids.shape
					if perp is None:
						if mode == 'pixel':
							perp_range = range(1, self.image_mids.swapaxes(2, -1).reshape(-1, orig_shape[2]).shape[0],
							                   int(
								                   np.around(
									                   self.image_mids.swapaxes(2, -1).reshape(-1, orig_shape[2]).shape[
										                   0] / 10)))
						else:
							perp_range = range(1, self.image_mids.swapaxes(2, 0).reshape(orig_shape[0] * orig_shape[2],
							                                                             -1).shape[1], int(
								np.around(
									self.image_mids.swapaxes(2, 0).reshape(orig_shape[0] * orig_shape[2], -1).shape[
										1] / 10)))
					else:
						perp_range = range(perp, perp + 1, 1)
					for perp in perp_range:
						# range(perp_range[mode][0], perp_range[mode][1], perp_range[mode][2]):
						fig = plt.figure()
						axis = fig.add_axes([0, 0, 1, 1])

						all_outputs = []  # np.zeros([image_mids.shape[0], img_size[0] * img_size[1]], dtype=np.float)

						orig_shape = self.image_mids.shape
						tsne = TSNE(mode, perp, n_iter=n_iter)
						try:
							if mode == 'pixel':
								# 	tsne.transform(image_mids[l, 0].reshape(shape[1], shape[2], shape[3]), save=False)
								tsne.transform(self.image_mids.swapaxes(2, -1).reshape(-1, orig_shape[2]))
							else:
								tsne.transform(
									self.image_mids.swapaxes(2, 0).reshape(orig_shape[0] * orig_shape[2], -1))

							if mode == 'pixel':
								outputs = tsne.outputs[0].reshape(orig_shape[0], orig_shape[1], orig_shape[3],
								                                  orig_shape[4], 2)
								for dom in range(0, orig_shape[0]):
									for i in range(0, 4):
										output = outputs[dom, self.eval_maps[dom] == i]
										outputs_subset = np.random.choice(range(len(output)),
										                                  min(desired_points, len(output)),
										                                  replace=False)
										if len(outputs_subset > 0):
											axis.scatter(x=output[outputs_subset, 0],
											             y=output[outputs_subset, 1],
											             label=self.names[dom] + ' ' + img_types[i], s=20,
											             alpha=0.7)

									all_outputs.append(outputs[dom])
							# all_outputs.append(np.concatenate([outputs[dom, self.eval_maps[dom] == 0], outputs[dom, self.eval_maps[dom] == 2]]))
							# all_outputs.append(np.concatenate([ outputs[dom, self.eval_maps[dom] == 0], outputs[dom, self.eval_maps[dom] == 2],
							#                                     outputs[dom, self.eval_maps[dom] == 1], outputs[dom, self.eval_maps[dom] == 3]]))

							else:
								# TODO: subset length is wrong
								outputs = tsne.outputs[0].reshape(orig_shape[0], orig_shape[2], -1)
								for dom in range(0, orig_shape[0]):
									outputs_subset = np.random.choice(range(len(outputs[dom])),
									                                  min(desired_points, len(outputs[dom])),
									                                  replace=False)
									axis.scatter(x=outputs[dom, outputs_subset][:, 0],
									             y=outputs[dom, outputs_subset][:, 1], label=self.names[dom], s=20,
									             alpha=0.7)

									all_outputs.append(outputs[dom, outputs_subset])
							# plt.scatter(outputs[0][:, 0], outputs[0][:, 1])
							# plt.show()
							axis.legend()
							if save:
								np.save("data/graphs/tsne/" + str(layer_name) + "_mode_" + str(
									tsne.mode) + "_thresh_" + str(
									threshold) + "_cut_" + str(mask_cut) + "_perp_" + str(
									tsne.perp) + str(img_size) + ".csv", np.array(all_outputs))
								fig.savefig(
									"data/graphs/tsne/" + str(layer_name) + "_mode_" + str(
										tsne.mode) + "_thresh_" + str(
										threshold) + "_cut_" + str(mask_cut) + "_perp_" + str(
										tsne.perp) + str(img_size) + ".jpg")
							plt.close(fig)

							self.output = np.array(all_outputs)
						except:
							pass
				if algo == 'pca':
					fig = plt.figure()
					axis = fig.add_axes([0, 0, 1, 1])
					all_outputs = []
					fit = True
					pca = PCA(mode, fit=fit, n_components=min(2, self.image_mids[0].shape[1]))
					for dom in range(0, self.image_mids.shape[0]):
						shape = self.image_mids[dom].shape
						# TODO: Check how it is better to feed samples, together or separately
						pca.transform(self.image_mids[dom].swapaxes(1, -1).reshape(-1, shape[1]))
						pca.fit = False
						outputs = np.array(pca.outputs[dom]).reshape(-1, 2)
						outputs_subset = np.random.choice(range(outputs.shape[0]),
						                                  min(desired_points, outputs.shape[0]),
						                                  replace=False)
						axis.scatter(x=outputs[outputs_subset][:, 0],
						             y=outputs[outputs_subset][:, 1], label=self.names[dom], s=20, alpha=0.7)
						# plt.scatter(x=outputs[outputs_subset][:, 0],
						#              y=outputs[outputs_subset][:, 1], label=self.names[dom], s=20, alpha=0.7)
						# plt.show()
						all_outputs.append(outputs[outputs_subset].reshape(shape[0], shape[2], shape[3], 2))
					axis.legend()

					if save:
						fig.savefig(
							"data/graphs/pca/" + str(layer_name) + "_mode_" + str(pca.mode) + "_thresh_" + str(
								threshold) + "_cut_" + str(mask_cut) + str(img_size) + ".jpg")
					plt.close(fig)
					self.output = np.array(all_outputs)

				if algo == 'isomap':
					isomap = Isomap(mode)
					isomap.transform(self.image_mids)

					fig = plt.figure()
					axis = fig.add_axes([0, 0, 1, 1])

					outputs = np.array(isomap.outputs).reshape(-1, 2)
					outputs_subset = np.random.choice(range(len(outputs)), min(desired_points, len(outputs)),
					                                  replace=False)
					axis.scatter(x=outputs[outputs_subset][:, 0],
					             y=outputs[outputs_subset][:, 1], label=self.names[l], s=20, alpha=0.7)
					axis.legend()
					fig.savefig(
						"data/graphs/isomap/" + str(layer_name) + "_mode_" + str(isomap.mode) + str(
							img_size) + ".jpg")
					plt.close(fig)

				if algo == 'lle':
					lle = LLE(mode)
					lle.transform(self.image_mids)

					fig = plt.figure()
					axis = fig.add_axes([0, 0, 1, 1])

					outputs = np.array(lle.outputs).reshape(-1, 2)
					outputs_subset = np.random.choice(range(len(outputs)), min(desired_points, len(outputs)),
					                                  replace=False)
					axis.scatter(x=outputs[outputs_subset][:, 0],
					             y=outputs[outputs_subset][:, 1], label=self.names[l], s=20, alpha=0.7)
					axis.legend()
					fig.savefig(
						"data/graphs/lle/" + str(layer_name) + "_mode_" + str(lle.mode) + str(
							img_size) + ".jpg")
					plt.close(fig)

				if algo == 'full':
					orig_shape = self.image_mids.shape
					if perp is None:
						if mode == 'pixel':
							perp_range = range(1, self.image_mids.swapaxes(2, -1).reshape(-1, orig_shape[2]).shape[0],
							                   int(
								                   np.around(
									                   self.image_mids.swapaxes(2, -1).reshape(-1, orig_shape[2]).shape[
										                   0] / 10)))
						else:
							perp_range = range(1, self.image_mids.swapaxes(2, 0).reshape(orig_shape[0] * orig_shape[2],
							                                                             -1).shape[1], int(
								np.around(
									self.image_mids.swapaxes(2, 0).reshape(orig_shape[0] * orig_shape[2], -1).shape[
										1] / 10)))
					else:
						perp_range = range(perp, perp + 1, 1)
					for perp in perp_range:
						# range(perp_range[mode][0], perp_range[mode][1], perp_range[mode][2]):
						fig = plt.figure()
						axis = fig.add_axes([0, 0, 1, 1])

						all_outputs = []  # np.zeros([image_mids.shape[0], img_size[0] * img_size[1]], dtype=np.float)

						orig_shape = self.image_mids.shape
						tsne = TSNE(mode, perp, n_iter=n_iter, init='pca')
						if mode == 'pixel':
							# 	tsne.transform(image_mids[l, 0].reshape(shape[1], shape[2], shape[3]), save=False)
							tsne.transform(self.image_mids.swapaxes(2, -1).reshape(-1, orig_shape[2]))
						else:
							tsne.transform(self.image_mids.swapaxes(2, 0).reshape(orig_shape[0] * orig_shape[2], -1))

						if mode == 'pixel':
							outputs = tsne.outputs[0].reshape(orig_shape[0], orig_shape[1], orig_shape[3],
							                                  orig_shape[4], 2)
							for dom in range(0, orig_shape[0]):
								for i in range(0, 4):
									output = outputs[dom, self.eval_maps[dom] == i]
									outputs_subset = np.random.choice(range(len(output)),
									                                  min(desired_points, len(output)),
									                                  replace=False)
									if len(outputs_subset > 0):
										axis.scatter(x=output[outputs_subset, 0],
										             y=output[outputs_subset, 1],
										             label=self.names[dom] + ' ' + img_types[i], s=20,
										             alpha=0.7)
								all_outputs.append(np.concatenate(
									[outputs[dom, self.eval_maps[dom] == 0], outputs[dom, self.eval_maps[dom] == 2]]))
						# all_outputs.append(np.concatenate([ outputs[dom, self.eval_maps[dom] == 0], outputs[dom, self.eval_maps[dom] == 2],
						#                                     outputs[dom, self.eval_maps[dom] == 1], outputs[dom, self.eval_maps[dom] == 3]]))
						else:
							# TODO: subset length is wrong
							outputs = tsne.outputs[0].reshape(orig_shape[0], orig_shape[2], -1)
							for dom in range(0, orig_shape[0]):
								outputs_subset = np.random.choice(range(len(outputs[dom])),
								                                  min(desired_points, len(outputs[dom])),
								                                  replace=False)
								axis.scatter(x=outputs[dom, outputs_subset][:, 0],
								             y=outputs[dom, outputs_subset][:, 1], label=self.names[dom], s=20,
								             alpha=0.7)

								all_outputs.append(outputs[dom, outputs_subset])

						axis.legend()
						if save:
							fig.savefig(
								"data/graphs/full/" + str(layer_name) + "_mode_" + str(tsne.mode) + "_thresh_" + str(
									threshold) + "_cut_" + str(mask_cut) + "_perp_" + str(
									tsne.perp) + str(img_size) + ".jpg")
						plt.close(fig)

						self.output = np.array(all_outputs)
				# for perp in range(perp_range[mode][0], perp_range[mode][1], perp_range[mode][2]):
				# 	fig = plt.figure()
				# 	axis = fig.add_axes([0, 0, 1, 1])
				# 	for l in range(0, self.image_mids.shape[0]):
				# 		shape = self.image_mids[l].shape
				# 		pca = PCA(mode)
				#
				# 		fit = True
				# 		# TODO check for right
				# 		mids = self.image_mids[l].reshape(shape[0] * shape[1], shape[2], shape[3])
				# 		for image_mid in mids:
				# 			image_mid = pca.get_manifold(image_mid, fit=fit)
				# 			fit = False
				#
				# 		tsne = TSNE(mode, perp)
				# 		tsne.transform(mids, save=False)  # !!
				#
				# 		outputs = np.array(tsne.outputs).reshape(-1, 2)
				# 		outputs_subset = np.random.choice(range(len(outputs)),
				# 		                                  min(desired_points, len(outputs)),
				# 		                                  replace=False)
				# 		axis.scatter(x=outputs[outputs_subset][:, 0],
				# 		             y=outputs[outputs_subset][:, 1], label=self.names[l], s=20, alpha=0.7)
				# 	axis.legend()
				# 	fig.savefig(
				# 		"data/graphs/full/" + str(layer_name) + "_mode_" + str(tsne.mode) + "_perp_" + str(
				# 			tsne.perp) + str(img_size) + ".jpg")
				#
				# 	plt.close(fig)

				if algo == 'pca_cuda':
					fig = plt.figure()
					axis = fig.add_axes([0, 0, 1, 1])
					all_outputs = []
					fit = True
					pca = PCA_cuda(mode, fit=fit, n_components=min(2, self.image_mids[0].shape[1]))
					for dom in range(0, self.image_mids.shape[0]):
						shape = self.image_mids[dom].shape
						# TODO: Check how it is better to feed samples, together or separately
						pca.transform(self.image_mids[dom].swapaxes(1, -1).reshape(-1, shape[1]))
						pca.fit = False
						outputs = np.array(pca.outputs[dom]).reshape(-1, 2)
						outputs_subset = np.random.choice(range(outputs.shape[0]),
						                                  min(desired_points, outputs.shape[0]),
						                                  replace=False)
						axis.scatter(x=outputs[outputs_subset][:, 0],
						             y=outputs[outputs_subset][:, 1], label=self.names[dom], s=20, alpha=0.7)
						# plt.scatter(x=outputs[outputs_subset][:, 0],
						#              y=outputs[outputs_subset][:, 1], label=self.names[dom], s=20, alpha=0.7)
						# plt.show()
						all_outputs.append(outputs[outputs_subset].reshape(shape[0], shape[2], shape[3], 2))
					axis.legend()

					if save:
						fig.savefig(
							"data/graphs/pca/" + str(layer_name) + "_mode_" + str(pca.mode) + "_thresh_" + str(
								threshold) + "_cut_" + str(mask_cut) + str(img_size) + ".jpg")
					plt.close(fig)
					self.output = np.array(all_outputs)

	def get_data(self):
		keys = ['input', 'output', 'names', 'eval_maps', 'image_mids']
		self.__dict__.pop('net', None)

		return self.__dict__

	@staticmethod
	def auto(algos, modes, layer_name, threshold, img_size, volume_num, sample_num, mask_cut,
	         perp, n_iter, save):
		reductor = Reductor()
		reductor.reduction(algos, modes, layer_name, threshold, img_size, volume_num, sample_num, mask_cut,
		                   perp, n_iter, save)

		return reductor



class MapPresenter():
	def __init__(self,
	             model_path="E:\Thesis\conp-dataset\projects\calgary-campinas\CC359\Train2\checkpoints\CP_epoch200.pth"):

		"""Parameters Init"""
		self.names = []

		self.image_mids = []
		self.eval_maps = []
		self.input = []
		self.output = []
		self.input_img_size = (128, 128)
		self.activ_img_size = (8, 8)
		self.coord = np.zeros(self.activ_img_size)

		""" Net setup """
		self.net = UNet2D(n_chans_in=1, n_chans_out=1)
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.net.to(device=device)
		self.net.load_state_dict(torch.load(model_path, map_location=device))

	def present(self, layer_name, img_size):

		self.activ_img_size = img_size

		data = get_data(volume_num=1, sample_num=1)

		self.names.append(data[0][1])
		test_loader = data[0][0]

		for batch in test_loader:
			image, true_mask = batch['image'], batch['mask']
			true_mask = torch.sigmoid(true_mask)
			self.true_mask = true_mask.cpu().numpy()[0, 0]

			self.image_mid, self.net_output = get_mid_output(self.net, image, layer_name)
			break

		n = int(np.ceil(np.sqrt(self.image_mid.shape[0])))
		# add epty cells to make it square
		self.image_mid = np.concatenate((self.image_mid,
		                np.zeros((n * n - self.image_mid.shape[0], self.image_mid.shape[1], self.image_mid.shape[2]))),
		               axis=0)
		self.image_mid = self.image_mid.reshape(n, n, self.image_mid.shape[1], self.image_mid.shape[2])

		x=0

	def get_data(self):
		keys = ['input', 'output', 'names', 'eval_maps', 'image_mids']
		self.__dict__.pop('net', None)

		return self.__dict__

	@staticmethod
	def auto(modes, layer_name, img_size, mask_cut):
		mapPresenter = MapPresenter()
		mapPresenter.present(layer_name, img_size)
		return mapPresenter



if __name__ == '__main__':
	""" Variables setup """
	layer = 'init_path'
	img_size = (128, 128)
	map = MapPresenter.auto(0,layer,img_size,0)

	# algos = ['pca_cuda']  # ['tsne', 'pca', 'isomap', 'lle', 'full']
	# modes = ['pixel', 'feature']  # ['feature', 'pixel','pic_to_coord']  # smth else?
	#
	# layers = ['init_path', 'down1', 'down2', 'up1', 'out_path']
	#
	# thresholds = [None]
	# img_size = (32, 32)
	# img_sizes = [(8, 8), (16, 16), (32, 32), (64, 64), (128, 128)]
	# show_im = False
	# crop_im = False
	# volume_num = 2
	# sample_num = 2
	# perp = 1
	# save = True
	#
	# mask_cuts = [None]  # 'true', 'predict', None
	# n_init = 250
	#
	# n_iters = [500]
	# """ Image processing and layer output """
	# for n_iter in n_iters:
	# 	for threshold in thresholds:
	# 		for mask_cut in mask_cuts:
	# 			for img_size in img_sizes:
	# 				for layer_name in layers:
	# 					Reductor.auto(algos, modes, layer_name, threshold, img_size, volume_num, sample_num, mask_cut,
	# 					              perp, n_iter, save)

