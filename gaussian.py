import numpy as np
from sklearn.mixture import GaussianMixture
import os
from os import walk
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

np.random.seed(0)


graphs_dir = "E:\\Thesis\\DomainVis\\reductor\\data\\graphs"
model_list = [
    'philips_15',
    'philips_3',
    'siemens_15',
    'siemens_3',
    'ge_15',
    'ge_3'
]
algo_list = [
    'full',
    'isomap',
    'lle',
    'pca',
    'tsne',
]
f = []
dic = {}
for algo in algo_list:

    path = os.path.join(graphs_dir, algo)
    filenames = next(walk(path), (None, None, []))[2]
    for file in filenames:
        if file.split('.')[1]=='npy':
            layer = filenames[0].split('_')[0]
            mode = filenames[0].split('_')[2]
            data = np.load(os.path.join(path,file)).reshape(6,-1,2)
            con_data = np.concatenate(data,axis=0)
            con_data_s = con_data

            y=data
            X=con_data_s


            default_base = {

                "n_clusters": 6,
                "quantile": 0.1,
                "eps": 0.3,
                "damping": 0.9,
                "preference": -200,
                "n_neighbors": 10,

                "min_samples": 20,
                "xi": 0.05,
                "min_cluster_size": 0.1,
            }



            params = default_base.copy()

            # normalize dataset for easier parameter selection
            X = StandardScaler().fit_transform(X)

            # estimate bandwidth for mean shift


            # connectivity matrix for structured Ward
            connectivity = kneighbors_graph(
                X, n_neighbors=params["n_neighbors"], include_self=False
            )
            # make connectivity symmetric
            connectivity = 0.5 * (connectivity + connectivity.T)

            # ============
            # Create cluster objects
            # ============

            bandwidth = cluster.estimate_bandwidth(X, quantile=params["quantile"])
            ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)


            two_means = cluster.MiniBatchKMeans(n_clusters=params["n_clusters"])
            ward = cluster.AgglomerativeClustering(
                n_clusters=params["n_clusters"], linkage="ward", connectivity=connectivity
            )
            spectral = cluster.SpectralClustering(
                n_clusters=params["n_clusters"],
                eigen_solver="arpack",
                affinity="nearest_neighbors",
            )
            dbscan = cluster.DBSCAN(eps=params["eps"])
            optics = cluster.OPTICS(
                min_samples=params["min_samples"],
                xi=params["xi"],
                min_cluster_size=params["min_cluster_size"],
            )
            affinity_propagation = cluster.AffinityPropagation(
                damping=params["damping"], preference=params["preference"], random_state=0
            )
            average_linkage = cluster.AgglomerativeClustering(
                linkage="average",
                affinity="cityblock",
                n_clusters=params["n_clusters"],
                connectivity=connectivity,
            )
            birch = cluster.Birch(n_clusters=params["n_clusters"])
            gmm = mixture.GaussianMixture(
                n_components=params["n_clusters"], covariance_type="full"
            )

            clustering_algorithms = (
                ("MiniBatch\nKMeans", two_means),
                ("Affinity\nPropagation", affinity_propagation),
                ("MeanShift", ms),
                ("Spectral\nClustering", spectral),
                ("Ward", ward),
                ("Agglomerative\nClustering", average_linkage),
                ("DBSCAN", dbscan),
                ("OPTICS", optics),
                ("BIRCH", birch),
                ("Gaussian\nMixture", gmm),
            )

            for name, algorithm in clustering_algorithms:
                t0 = time.time()

                # catch warnings related to kneighbors_graph
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="the number of connected components of the "
                        + "connectivity matrix is [0-9]{1,2}"
                        + " > 1. Completing it to avoid stopping the tree early.",
                        category=UserWarning,
                    )
                    warnings.filterwarnings(
                        "ignore",
                        message="Graph is not fully connected, spectral embedding"
                        + " may not work as expected.",
                        category=UserWarning,
                    )
                    algorithm.fit(X)

                t1 = time.time()
                if hasattr(algorithm, "labels_"):
                    y_pred = algorithm.labels_.astype(int)
                else:
                    y_pred = algorithm.predict(X)

                plt.title(name, size=18)

                colors = np.array(
                    list(
                        islice(
                            cycle(
                                [
                                    "#377eb8",
                                    "#ff7f00",
                                    "#4daf4a",
                                    "#f781bf",
                                    "#a65628",
                                    "#984ea3",
                                    "#999999",
                                    "#e41a1c",
                                    "#dede00",
                                ]
                            ),
                            int(max(y_pred) + 1),
                        )
                    )
                )
                # add black color for outliers (if any)
                colors = np.append(colors, ["#000000"])
                plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

                plt.xlim(-2.5, 2.5)
                plt.ylim(-2.5, 2.5)
                plt.xticks(())
                plt.yticks(())
                plt.text(
                    0.99,
                    0.01,
                    ("%.2fs" % (t1 - t0)).lstrip("0"),
                    transform=plt.gca().transAxes,
                    size=15,
                    horizontalalignment="right",
                )

            plt.show()
            