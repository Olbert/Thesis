import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import torch
import sklearn.manifold
import matplotlib.pyplot as plt
from torch.optim import Adam
from torchvision import models
from sklearn import preprocessing
from DomainVis.unet.model import UNet2D
from DomainVis.unet.misc_functions import preprocess_image, recreate_image, save_image
from PIL import Image
from DomainVis.database_process.dataset import BasicDataset
from sklearn.decomposition import PCA
import openTSNE
from openTSNE import affinity, initialization
scaler = preprocessing.MinMaxScaler()
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
import skcuda.linalg as linalg
from skcuda.linalg import PCA as cuPCA


from matplotlib import pyplot as plt
from sklearn import datasets

class Basic_reduction():
	def __init__(self, mode):
		self.mode = mode
		self.function = None
		self.outputs = []

	def get_manifold(self, output_mid, fit=False):  # TODO: check function params

		if (self.mode == 'pixel'):
			if fit:
				output_2d = self.function.fit_transform(output_mid).astype(
					np.float)
			else:
				output_2d = self.function.transform(output_mid).astype(
					np.float)



		elif (self.mode == 'feature'):
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

	def transform(self, images):
		self.outputs.append([self.get_manifold(images, fit=self.fit)])
		return self.outputs


class PCA(Basic_reduction):

	def __init__(self, mode, n_components=2, fit=True):
		super().__init__(mode)
		self.fit = fit
		self.function = sklearn.decomposition.PCA(n_components=n_components)

class PCA_cuda(Basic_reduction):

	def __init__(self, mode, n_components=2, fit=True):
		super().__init__(mode)
		self.fit = fit
		self.function = cuPCA(n_components=n_components)

	def transform(self, images):

		X = np.asfortranarray(images)
		i=0
		precisions = ['double']

		X_gpu = gpuarray.to_gpu(X)  # copy data to gpu

		T_gpu = self.function.fit_transform(X_gpu)  # calculate the principal components

		# show that the resulting eigenvectors are orthogonal
		# Note that GPUArray.copy() is necessary to create a contiguous array
		# from the array slice, otherwise there will be undefined behavior
		dot_product = linalg.dot(T_gpu[:, 0].copy(), T_gpu[:, 1].copy())
		T = T_gpu.get()

		print("The dot product of the first two " + str(precisions[i]) +
		      " precision eigenvectors is: " + str(dot_product))

		# now get the variance of the eigenvectors and create the ratio explained from the total
		std_vec = np.std(T, axis=0)
		print("We explained " + str(100 * np.sum(std_vec[:2]) / np.sum(std_vec)) +
		      "% of the variance with 2 principal components in " +
		      str(precisions[i]) + " precision\n\n")



class LLE(Basic_reduction):

	def __init__(self, mode, n_components=2):
		super().__init__(mode)

		self.function = sklearn.manifold.LocallyLinearEmbedding(n_components=n_components)



class Isomap(Basic_reduction): # TODO: Not done (n_neighbours is always 2

	def __init__(self, mode, n_components=2):
		super().__init__(mode)

		self.function = sklearn.manifold.Isomap(n_components=n_components,n_neighbors=2)


class TSNE(Basic_reduction):  # TODO static?

	def __init__(self, mode, perp, init = 'random', n_components=2, n_iter=500):
		super().__init__(mode)
		self.perp = perp
		self.mode = mode
		self.init = init
		# self.function = sklearn.manifold.TSNE(n_components=n_components, perplexity=perp, init=init, n_iter=n_iter)
		self.function = openTSNE.TSNE(
			perplexity=30,
		    initialization="pca",
		    metric="cosine",
		    n_jobs=8,
		    random_state=3,
		)

	def get_manifold(self, output_mid, fit=False):
		if (self.mode == 'pixel'):
			# output_mid = output_mid.reshape(output_mid.shape[0], -1).swapaxes(0, 1)

			affinities_multiscale_mixture = openTSNE.affinity.Multiscale(
				output_mid,
				perplexities=[50, 1000],
				metric="cosine",
				n_jobs=8,
				random_state=3,
			)
			init = openTSNE.initialization.pca(output_mid, random_state=42)

			embedding_multiscale = openTSNE.TSNE(n_jobs=8).fit(
				affinities=affinities_multiscale_mixture,
				initialization=init,
			)
			output_2d = embedding_multiscale

			# output_2d = self.function.fit_transform(output_mid).astype(
				# 	np.float)


		elif (self.mode == 'feature'):
			# output_mid = output_mid.reshape(output_mid.shape[0],-1)

			# output_2d = self.function.fit_transform(output_mid).astype(
			# 			# 	np.float)
			affinities_multiscale_mixture = openTSNE.affinity.Multiscale(
				output_mid,
				perplexities=[2, 20],
				metric="cosine",
				n_jobs=8,
				random_state=3,
			)
			init = openTSNE.initialization.pca(output_mid, random_state=42)

			embedding_multiscale = openTSNE.TSNE(n_jobs=8).fit(
				affinities=affinities_multiscale_mixture,
				initialization=init,
			)
			output_2d = embedding_multiscale

		elif (self.mode == 'pic_to_coord'):
			output_mid = output_mid.reshape(1, -1)

			output_2d = self.function.fit_transform(output_mid).astype(
				np.float)

		output_2d = scaler.fit_transform(output_2d)  # ?

		return output_2d


	def transform(self, images):
		self.outputs.append(self.get_manifold(images, fit=True))
		return self.outputs
class Sammon():
	zz = 0
