import seaborn as sns
import umap
import matplotlib.pyplot as plt
from sklearn.cluster import *
import torch
#from sklearn_extra.cluster import KMedoids
import numpy as np
import hdbscan
from munkres import Munkres,print_matrix
from sklearn.mixture import GaussianMixture
def error(cluster, target_cluster, k):
    """ Compute error between cluster and target cluster
    :param cluster: proposed cluster
    :param target_cluster: target cluster
    :return: error
    """
    n = np.shape(target_cluster)[0]
    M = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            M[i][j] = np.sum(np.logical_and(cluster == i, target_cluster == j))
    m = Munkres()
    indexes = m.compute(-M)
    corresp = []
    for i in range(k):
        corresp.append(indexes[i][1])
    pred_corresp = [corresp[int(predicted)] for predicted in cluster]
    acc = np.sum(pred_corresp == target_cluster) / float(len(target_cluster))
    return acc
numpy_array = np.load('feature.npy')
labels = np.load('label.npy')

clusterable_embedding = umap.UMAP(
    n_neighbors=30,
    min_dist=0.0,
    n_components=2,
    random_state=42,
).fit_transform(numpy_array)

clusterer = hdbscan.HDBSCAN(min_samples=10,min_cluster_size=140)
#clusterer.fit(clusterable_embedding)
#clusterer = KMedoids(n_clusters=10, random_state=0)
#clusterer =KMeans(n_clusters=10)
#clusterer =AffinityPropagation(damping=0.8,preference=-14,random_state=0).fit(numpy_array)
#clusterer = GaussianMixture(n_components=10)
clusterer.fit(clusterable_embedding)
#clusterer.labels_= clusterer.predict(numpy_array)
#clusterer.fit(clusterable_embedding)
x=error(labels,clusterer.labels_,10)
print("正确率:",x)
print("聚类数:",clusterer.labels_.max()+1)
from sklearn.metrics import *
score_sil=silhouette_score(clusterable_embedding,clusterer.labels_)
NMI=normalized_mutual_info_score(labels,clusterer.labels_)
ARI=adjusted_rand_score(labels,clusterer.labels_)
print("ARI:",ARI)
print("NMI:",NMI)
print("轮廓系数:",score_sil)
palette = sns.color_palette()
#cluster_colors = [sns.desaturate(palette[col], sat)
                  #if col >= 0 else (0.5, 0.5, 0.5) for col, sat in
                  #zip(clusterer.labels_, clusterer.probabilities_)]
plt.figure(dpi=300)
plt.scatter(clusterable_embedding[:, 0], clusterable_embedding[:, 1],c=clusterer.labels_,  s=0.1, cmap='Spectral')
plt.show()