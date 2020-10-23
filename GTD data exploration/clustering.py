from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
import feature_selection
import matplotlib.pyplot as plt
import numpy as np



# def k_means_explore(X):
#     wcss = []
#     for i in range(1, 11):
#         kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
#         kmeans.fit(X)
#         wcss.append(kmeans.inertia_)
#     plt.plot(range(1, 11), wcss)
#     plt.title('Elbow Method')
#     plt.xlabel('Number of clusters')
#     plt.ylabel('WCSS')
#     plt.show()
#
# def k_prototypes(X, n):
#     kproto=KPrototypes(n_clusters=n)
#     cat_cols=feature_selection.categorical_number_columns+feature_selection.categorical_string_cols
#     cols = [X.columns.get_loc(c) for c in cat_cols if c in X]
#     pred_y=kproto.fit_predict(X, categorical=cols)
#     return pred_y

def hierarchical_clustering(X):
    Z = linkage(X, 'ward')
    c, coph_dists = cophenet(Z, pdist(X))
    print("c: ", c)
    evaluate(Z)

def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        # plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('Cluster size')
        plt.ylabel('Distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

def evaluate(Z):
    fancy_dendrogram(
        Z,
        truncate_mode='lastp',
        p=12,
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,
        annotate_above=10,  # useful in small plots so annotations don't overlap
    )
    last = Z[-10:, 2]
    last_rev = last[::-1]
    idxs = np.arange(1, len(last) + 1)
    # plt.plot(idxs, last_rev)

    acceleration = np.diff(last, 2)  # 2nd derivative of the distances
    acceleration_rev = acceleration[::-1]
    # plt.plot(idxs[:-2] + 1, acceleration_rev)
    # plt.show()
    k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
    print("clusters: ", k)
    plt.show()
