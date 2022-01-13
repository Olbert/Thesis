import numpy as np
import numpy as np
from sklearn.mixture import GaussianMixture
import os
from os import walk
import time
import warnings
import scipy
from scipy.special import rel_entr
import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

from scipy.spatial.distance import jensenshannon

from itertools import cycle, islice
import matplotlib.pyplot as plt
np.random.seed(0)


def KLdivergence(x, y):
	"""Compute the Kullback-Leibler divergence between two multivariate samples.
  Parameters
  ----------
  x : 2D array (n,d)
    Samples from distribution P, which typically represents the true
    distribution.
  y : 2D array (m,d)
    Samples from distribution Q, which typically represents the approximate
    distribution.
  Returns
  -------
  out : float
    The estimated Kullback-Leibler divergence D(P||Q).
  References
  ----------
  Pérez-Cruz, F. Kullback-Leibler divergence estimation of
continuous distributions IEEE International Symposium on Information
Theory, 2008.
  """
	from scipy.spatial import cKDTree as KDTree

	# Check the dimensions are consistent
	x = np.atleast_2d(x)
	y = np.atleast_2d(y)

	n, d = x.shape
	m, dy = y.shape

	assert (d == dy)

	# Build a KD tree representation of the samples and find the nearest neighbour
	# of each point in x.
	xtree = KDTree(x)
	ytree = KDTree(y)

	# Get the first two nearest neighbours for x, since the closest one is the
	# sample itself.
	r = xtree.query(x, k=2, eps=.01, p=2)[0][:, 1]
	s = ytree.query(x, k=1, eps=.01, p=2)[0]

	# There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
	# on the first term of the right hand side.
	return -np.log(r / s).sum() * d / n + np.log(m / (n - 1.))


graphs_dir = "E:\\Thesis\\DomainVis\\reductor\\data\\graphs"
names = [
	'philips_15',
	'philips_3',
	'siemens_15',
	'siemens_3',
	'ge_15',
	'ge_3'
]
algo_list = [
	# 'full',
	'isomap',
	'lle',
	'pca',
	'tsne',
]
# algo_list=['isomap','lle']
layer_list = ['init_path','down1','down2','down3','up1','up2']

# layer_list = ['down1']
f = []
dic = {}
# ids = [[3,2],[3,1]]
kls = np.zeros((len(algo_list), len(layer_list), len(names)))
a = 0
for algo in algo_list:
	z=0

	psi = np.zeros((len(layer_list), len(names) + 1, len(names)))
	path = os.path.join(graphs_dir, algo)
	filenames = next(walk(path), (None, None, []))[2]
	for file in filenames:
		if file.split('.')[1] == 'npy':
			layer = file.split('_')[0]
			# mode = file.split('_')[2]
			if file.split('_')[2]=='pixel' or file.split('_')[3]=='pixel':
				if layer in layer_list:
					z= layer_list.index(layer)
					# for id in ids:
					data = np.load(os.path.join(path, file)).reshape(6, -1, 2)


					for i in range(0, len(names)):
						for k in range(0, len(names)):
							psi[z,i,k] = KLdivergence(data[i], data[k])

							if i==3:
								kls[a, z, k] = np.clip(KLdivergence(data[i], data[k]), 0, 10)
							# plt.scatter(data[id[0]][:,0],data[id[0]][:,1])
							# plt.scatter(data[id[1]][:,0],data[id[1]][:,1])
							# plt.show()

	a += 1
for l in range(len(layer_list)):
	for a in range(len(algo_list)):
		norm = np.linalg.norm(kls[a, l,:])
		normal_array = kls[a, l,:] / norm
		plt.plot([*range(6)],  normal_array,label=algo_list[a])

	plt.plot([*range(6)],[0.02569609,	0.484416,	0.09828225,	0,	0.24960016,	0.00139129],label='distance',linewidth=3.0)
	plt.legend(loc="upper right")
	plt.xticks([*range(6)], names)
	plt.title(layer_list[l])
	plt.show()
	plt.clf()

x = 1+1
	#
	# np.set_printoptions(precision=2)
	# np.savetxt(algo+'.csv', np.concatenate(psi,axis=0), fmt="%8.6s")
	#






"""
[ 8.51548975 12.77323463]
[8.51548975 8.51548975]
[12.77323463 12.77323463]
[8.51548975 8.51548975]


[ 8.51548975 12.77323463]
[12.77323463  8.51548975]
[4.25774488 4.25774488]
[12.77323463 12.77323463]
[12.77323463  8.51548975]


12.772169859028775
"""
