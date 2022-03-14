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

from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
import numpy as np
import matplotlib.pylab as pl
import ot
import ot.plot

import matplotlib.pyplot as plt
np.random.seed(0)

def compute_kl_divergence1(p_probs, q_probs):
    """"KL (p || q)"""
    kl_div = 0.0
    for p, q in zip(p_probs, q_probs):
        kl_div += p * np.log(p / q)

    return kl_div


def compute_kl_divergence(p_probs, q_probs):
    """"KL (p || q)"""
    kl_div = p_probs * np.log(p_probs / q_probs)
    return np.sum(kl_div)

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
    return -np.log(r/ s).sum() * d / n + np.log(m / (n - 1.))

def empirical_dist(Xs,Xt):
    M = ot.dist(Xs, Xt, metric='sqeuclidean')
    M /= M.max()

    # EMD Transport
    # ot_emd = ot.da.EMDTransport()
    # ot_emd.fit(Xs=Xs, Xt=Xt)
    #
    # # Sinkhorn Transport
    # ot_sinkhorn = ot.da.SinkhornTransport(reg_e=1e-1)
    # ot_sinkhorn.fit(Xs=Xs, Xt=Xt)
    #
    #
    # # transport source samples onto target samples
    # transp_Xs_emd = ot_emd.transform(Xs=Xs)
    # transp_Xs_sinkhorn = ot_sinkhorn.transform(Xs=Xs)
    n = Xs.shape[0]

    n_seed = 50


    res = ot.sliced_wasserstein_distance(Xs, Xt)

    return res



def calculate(algo_list, layer_list,domains,base, graphs_dir):
    a = 0

    kls = np.zeros((len(algo_list), len(layer_list), len(domains)))

    for algo in algo_list:
        z = 0
        psi = np.zeros((len(layer_list), len(domains) + 1, len(domains)))
        path = os.path.join(graphs_dir, algo)
        filedomains = next(walk(path), (None, None, []))[2]
        for file in filedomains:
            if file.split('.')[1] == 'npy':
                layer = file.split('_')[0]
                # mode = file.split('_')[2]

                if file.split('_')[2] == 'pixel' or file.split('_')[3] == 'pixel':
                    if layer in layer_list:
                        z = layer_list.index(layer)
                        # for id in ids:
                        data = np.load(os.path.join(path, file)).reshape(6, -1, 2)
                        for k in range(0, len(domains)):
                                kls[a, z, k] = empirical_dist(data[base], data[k])


        a += 1

    np.save('kls', kls)
    return kls

def calculate_param(param_list, layer_list,domains,base,path, out_name):

    kls = np.zeros((len(param_list), len(layer_list), len(domains)))

    filenames = next(walk(path), (None, None, []))[2]

    for file in filenames:

        if file.split('.')[1] == 'npy':
            layer = file.split('_')[0]
            mode = file.split('_')[2]
            try:
                param = int((file.split('_')[8]).split('(')[0])
            except:
                param = int((file.split('_')[9]).split('(')[0])
            param_id = param_list.index(int(param))
            if file.split('_')[2] == 'pixel' or file.split('_')[3] == 'pixel':

                if layer in layer_list:
                    z = layer_list.index(layer)
                    # for id in ids:
                    data = np.load(os.path.join(path, file)).reshape(6, -1, 2)
                    for k in range(0, len(domains)):
                        kls[param_id, z, k] =empirical_dist(data[base], data[k])
                        # kl1 = KLdivergence(data[base], data[k])
                        # kl2 = KLdivergence(data[k],data[base],)
                        # if kl1 == 0 or kl1 == np.inf or kl1 == -np.inf:
                        #     if np.isnan(kl2):
                        #         kls[param_id, z, k] = kl1
                        #     else:
                        #         kls[param_id, z, k] = kl2
                        # elif np.isnan(kl1):
                        #    if np.isnan(kl2):
                        #        kls[param_id, z, k] = 0
                        #    else:
                        #         kls[param_id, z, k] = kl2
                        # else:
                        #     kls[param_id, z, k] = kl1
                        # kls[param_id, z, k] = np.clip(kls[param_id, z, k],0.0001,50)
    np.save(out_name, kls)
    return kls



def distance_plot_all():
    kls = np.load('kls.npy')
    kls = np.delete(kls, 3,2)
    dist = [0.02569609, 0.484416, 0.09828225, 0.24960016, 0.00139129]
    for l in range(len(layer_list)):
        for a in range(len(algo_list)):
            x = kls[a, l]
            normal_array = (x-min(x))/(max(x)-min(x)+0.00001)
            plt.plot([*range(kls.shape[2])],  normal_array,label=str(algo_list[a]))

        plt.plot([*range(kls.shape[2])],dist,label='distance',linewidth=3.0)

        plt.legend(loc="upper right")
        plt.xticks([*range(kls.shape[2])], domains_nosource)
        plt.title(layer_list[l])
        plt.show()
        plt.clf()

def distance_plot_algo(algo):
    kls = np.load('kls_'+algo+'.npy')
    kls = np.delete(kls, 3,2)
    dist = np.array([0.02569609, 0.484416, 0.09828225, 0.24960016, 0.00139129])/0.4
    # dist = (dist - min(dist)) / (max(dist) - min(dist) + 0.00001)
    for l in range(len(layer_list)):
        for a in range(len(param_list)):
            x = kls[a, l]*10
            # x = (x-min(x))/(max(x)-min(x)+0.00001)
            plt.plot([*range(kls.shape[2])],  x,label=str(param_list[a]))

        plt.plot([*range(kls.shape[2])],dist,label='distance',linewidth=3.0)

        plt.legend(loc="upper right")
        plt.xticks([*range(kls.shape[2])], domains_nosource)
        plt.title(layer_list[l])
        plt.show()
        plt.clf()

def cosine_matrix(kls, param_list):
    # kls = np.load('kls.npy')
    cosine = np.zeros((len(param_list), len(layer_list)))

    kls = np.delete(kls, 3, 2)
    dist = np.array(([0.02569609, 0.484416, 0.09828225, 0.24960016, 0.00139129]))
    dist = (dist - min(dist)) / (max(dist) - min(dist) + 0.00001)
    for l in range(len(layer_list)):
        for a in range(len(param_list)):
            x = kls[a, l]
            if x.sum() == 0:
                cosine[a, l] = np.nan
            else:
                normal_array = (x - min(x)) / (max(x) - min(x) + 0.00001)
                cosine[a,l] = 1 - spatial.distance.cosine(normal_array, dist)


    np.savetxt('cosine_isomap.csv',np.round(cosine,2),delimiter=', ', fmt="%.2f")
    return cosine





graphs_dir = "E:\\Thesis\\DomainVis\\reductor\\data\\graphs"
domains = [
    'philips_15',
    'philips_3',
    'siemens_15',
    'siemens_3',
    'ge_15',
    'ge_3'
]
domains_nosource = [
    'philips_15',
    'philips_3',
    'siemens_15',
    # 'siemens_3',
    'ge_15',
    'ge_3'
]
algo_list = [
    # 'full',
    # 'isomap',
    'lle',
    # 'pca',
    # 'tsne',
]
# algo_list=['isomap','lle']
layer_list = ['init','down1','down2','down3','up3','up2','up1']
# layer_list = ['init']

kls = np.zeros((len(algo_list), len(layer_list), len(domains)))

# [50, 100, 200, 5, 10, 30, 3, 25, 80]
param_list = [3, 5, 10, 30, 50, 100]
# calculate(algo_list,layer_list,domains,graphs_dir)
# kls = np.load('kls.npy')


# algo = 'isomap'
# # kls = calculate_param(param_list,
# #                 layer_list,
# #                 domains,
# #                 domains.index('siemens_3'),
# #                 os.path.join(graphs_dir, algo),
# #                 'kls_'+algo)
# kls = np.load('kls_'+algo+'.npy')
# distance_plot_algo(algo)
#
# cosine = cosine_matrix(kls)


# kls = calculate(algo_list,
#                 layer_list,
#                 domains,
#                 domains.index('siemens_3'),
#                 os.path.join(graphs_dir))

kls = np.load('kls.npy')
# distance_plot_all()
#
# cosine = cosine_matrix(kls,algo_list)
# idx = [0,1,2,3,6,5,4]
# for algo in algo_list:
#     for l in range(len(layer_list)):
#         x = kls[algo_list.index(algo), l]
#         kls[algo_list.index(algo), l] = (x - min(x)) / (max(x) - min(x) + 0.00001)
#     np.savetxt('kls_'+algo+'_norm.csv',np.round(kls[algo_list.index(algo)],2)[idx],delimiter=', ', fmt="%.2f")




algo = 'lle'
kls = calculate_param(param_list,
                layer_list,
                domains,
                domains.index('siemens_3'),
                os.path.join(graphs_dir, algo+'_all'),
                'kls_'+algo)
kls = np.load('kls_'+algo+'.npy')
distance_plot_algo(algo)


cosine = cosine_matrix(kls,param_list)

# name = os.path.join("E:\\Thesis\\DomainVis\\reductor\\data\\graphs\\pca",
#                     "init_path_mode_pixel_thresh_None_cut_None(128, 128).npy")
# data = np.load(name)
# print("Mean:")
# for i in range(data.shape[0]):
#     print(np.round(data[i].mean(),4))
# print("Var:")
# for i in range(data.shape[0]):
#     print(np.round(data[i].var(),4))
x= 1
x= x+1