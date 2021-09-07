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
from sklearn.decomposition import PCA

scaler = preprocessing.MinMaxScaler()


class Basic_reduction():
	def __init__(self, mode):
		self.mode = mode
		self.function = None
		self.fig = None
		self.outputs = []

	def get_manifold(self, output_mid, fit=False):  # TODO: check function params

		if (self.mode == 'pixel'):
			output_mid = output_mid.reshape(output_mid.shape[0], -1).swapaxes(0, 1)
			if fit:
				output_2d = self.function.fit_transform(output_mid).astype(
					np.float)
			else:
				output_2d = self.function.transform(output_mid).astype(
					np.float)



		elif (self.mode == 'feature'):

			output_mid = output_mid.reshape(output_mid.shape[0], -1)
			if fit:
				output_2d = self.function.fit_transform(output_mid).astype(
				np.float)
			else:
				output_2d = self.function.transform(output_mid).astype(
				np.float)


		elif (self.mode == 'pic_to_coord'):

			output_mid = output_mid.reshape(1, -1)
			if fit:
				output_2d = self.function.fit_transform(output_mid).astype(
					np.float)
			else:
				output_2d = self.function.transform(output_mid).astype(
					np.float)

		output_2d = scaler.fit_transform(output_2d)  # ?


		return output_2d

	def transform(self, images, save=True, folder="data/graphs/"):

		self.fig = plt.figure()
		axis = self.fig.add_axes([0, 0, 1, 1])

		output = self.get_manifold(images, fit=True)
		axis.scatter(x=output[:, 0], y=output[:, 1])
		self.outputs.append(output)

		# for i in range(1,images.shape[0]):
		# 	output = self.get_manifold(images[i], fit=False)
		# 	axis.scatter(x=output[:, 0], y=output[:, 1])
		# 	self.outputs.append(output)


class PCA(Basic_reduction):

	def __init__(self, mode, n_components=2):
		super().__init__(mode)

		self.function = sklearn.decomposition.PCA(n_components=n_components)

	def transform(self, images, save=True, folder="data/graphs/pca/"):
		super().transform(images, save, folder)
		if save:
			self.fig.savefig(folder  + "mode_" + str(self.mode) + ".jpg")
		plt.close(self.fig)
		return self.outputs
	# plt.show()


class LLE(Basic_reduction):

	def __init__(self, mode, n_components=2):
		super().__init__(mode)

		self.function = sklearn.manifold.LocallyLinearEmbedding(n_components=n_components)

	def transform(self, images, save=True, folder="data/graphs/lle/"):
		super().transform(images, save, folder)
		if save:
			self.fig.savefig(folder + "mode_" + str(self.mode) + ".jpg")
		plt.close(self.fig)
		return self.outputs
	# plt.show()


class Isomap(Basic_reduction): # TODO: Not done (n_neighbours is always 2

	def __init__(self, mode, n_components=2):
		super().__init__(mode)

		self.function = sklearn.manifold.Isomap(n_components=n_components,n_neighbors=2)

	def transform(self, images, save=True, folder="data/graphs/isomap/"):
		super().transform(images, save, folder)
		if save:
			self.fig.savefig(folder + "mode_" + str(self.mode) + ".jpg")
		plt.close(self.fig)
		return self.outputs
	# plt.show()


class TSNE(Basic_reduction):  # TODO static?

	def __init__(self, mode, perp, init = 'random', n_components=2, n_iter=500):
		super().__init__(mode)
		self.perp = perp
		self.mode = mode
		self.init = init
		self.function = sklearn.manifold.TSNE(n_components=n_components, perplexity=perp, init=init, n_iter=n_iter)

	def get_manifold(self, output_mid, fit=False):
		if (self.mode == 'pixel'):
			# output_mid = output_mid.reshape(output_mid.shape[0], -1).swapaxes(0, 1)

			output_2d = self.function.fit_transform(output_mid).astype(
				np.float)


		elif (self.mode == 'feature'):
			# output_mid = output_mid.reshape(output_mid.shape[0],-1)

			output_2d = self.function.fit_transform(output_mid).astype(
				np.float)

		elif (self.mode == 'pic_to_coord'):
			output_mid = output_mid.reshape(1, -1)

			output_2d = self.function.fit_transform(output_mid).astype(
				np.float)

		output_2d = scaler.fit_transform(output_2d)  # ?

		return output_2d

	def transform(self, images, save=True, folder="data/graphs/tsne/"):
		super().transform(images, save, folder)

		if save:
			self.fig.savefig(folder + "mode_" + str(self.mode) + "_perp_" + str(self.perp) + ".jpg")
		plt.close(self.fig)
		return self.outputs


class Sammon():
	zz = 0
