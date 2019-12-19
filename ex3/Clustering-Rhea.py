import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import numpy.linalg as linalg

# I don't use these in the code, imported for comparisons
from scipy.spatial.distance import cdist
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import kneighbors_graph


def circles_example():
    """
    an example function for generating and plotting synthetic data.
    """

    t = np.arange(0, 2 * np.pi, 0.03)
    length = np.shape(t)
    length = length[0]
    circle1 = np.array([np.cos(t) + 0.1 * np.random.randn(length),
                         np.sin(t) + 0.1 * np.random.randn(length)])
    circle2 = np.array([2 * np.cos(t) + 0.1 * np.random.randn(length),
                         2 * np.sin(t) + 0.1 * np.random.randn(length)])
    circle3 = np.array([3 * np.cos(t) + 0.1 * np.random.randn(length),
                         3 * np.sin(t) + 0.1 * np.random.randn(length)])
    circle4 = np.array([4 * np.cos(t) + 0.1 * np.random.randn(length),
                         4 * np.sin(t) + 0.1 * np.random.randn(length)])
    # circles = np.concatenate((circle1, circle2, circle3, circle4), axis=1)
    return np.concatenate((circle1, circle2, circle3, circle4), axis=1).T
    # plt.plot(circles[0,:], circles[1,:], '.k')
    # plt.show()


def apml_pic_example(path='APML_pic.pickle'):
    """
    an example function for loading and plotting the APML_pic data.

    :param path: the path to the APML_pic.pickle file
    """

    with open(path, 'rb') as f:
        apml = pickle.load(f)

    plt.plot(apml[:, 0], apml[:, 1], '.')
    plt.show()


def microarray_exploration(data_path='microarray_data.pickle',
                            genes_path='microarray_genes.pickle',
                            conds_path='microarray_conds.pickle'):
    """
    an example function for loading and plotting the microarray data.
    Specific explanations of the data set are written in comments inside the
    function.

    :param data_path: path to the data file.
    :param genes_path: path to the genes file.
    :param conds_path: path to the conds file.
    """

    # loading the data:
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    with open(genes_path, 'rb') as f:
        genes = pickle.load(f)
    with open(conds_path, 'rb') as f:
        conds = pickle.load(f)

    # look at a single value of the data matrix:
    i = 128
    j = 63
    print("gene: " + str(genes[i]) + ", cond: " + str(
        conds[0, j]) + ", value: " + str(data[i, j]))

    # make a single bar plot:
    plt.figure()
    plt.bar(np.arange(len(data[i, :])) + 1, data[i, :])
    plt.show()

    # look at two conditions and their correlation:
    plt.figure()
    plt.scatter(data[:, 27], data[:, 29])
    plt.plot([-5,5],[-5,5],'r')
    plt.show()

    # see correlations between conds:
    correlation = np.corrcoef(np.transpose(data))
    plt.figure()
    plt.imshow(correlation, extent=[0, 1, 0, 1])
    plt.colorbar()
    plt.show()

    # look at the entire data set:
    plt.figure()
    plt.imshow(data, extent=[0, 1, 0, 1], cmap="hot", vmin=-3, vmax=3)
    plt.colorbar()
    plt.show()


def euclid(X, Y):
    """
    return the pair-wise euclidean distance between two data matrices.
    :param X: NxD matrix.
    :param Y: MxD matrix.
    :return: NxM euclidean distance matrix.
    """
    # (x-y)^2=x^2+y^2-2xy
    X_squared_sum = np.sum(np.square(X), axis=1)  # ||X||^2
    Y_squared_sum = np.sum(np.square(Y), axis=1)  # ||Y||^2
    dists = -2 * np.dot(X, Y.T) + X_squared_sum[:, np.newaxis] + Y_squared_sum
    dists = np.clip(dists, 0, None)
    dists = np.sqrt(dists)
    return dists


def euclidean_centroid(X):
    """
    return the center of mass of data points of X.
    :param X: a sub-matrix of the NxD data matrix that defines a cluster.
    :return: the centroid of the cluster.
    """
    # the centroid of the cluster is the average of all points in the cluster
    num_points, _ = X.shape
    return X.sum(axis=0) / num_points


def kmeans_pp_init(X, k, metric):
    """
    The initialization function of kmeans++, returning k centroids.
    :param X: The data matrix.
    :param k: The number of clusters.
    :param metric: a metric function like specified in the kmeans documentation.
    :return: kxD matrix with rows containing the centroids.
    """
    samples, D = X.shape
    if k > samples:
        raise Exception("k needs to be smaller than the number of points in the data")
    centroids = np.zeros((k, D))
    random_row = np.random.randint(samples)
    centroids[0, :] = X[random_row, :]  # initialize the first centroid at random
    for i in range(1, k):
        # define a probability vector s.t. p()
        # compute distances of X and all centroids
        dist = metric(X, centroids[:i, :])  # a samples x i matrix
        dist = np.min(dist, axis=1)  # get the minimal distance from some centroid
        sq_dist = np.square(dist)  # the probability is proportionate to the squared distance
        p = sq_dist / np.sum(sq_dist)
        centroids[i, :] = X[np.random.choice(samples, 1, p=p)]
    return centroids


def kmeans(X, k, iterations=10, metric=euclid, center=euclidean_centroid, init=kmeans_pp_init):
    """
    The K-Means function, clustering the data X into k clusters.
    :param X: A NxD data matrix.
    :param k: The number of desired clusters.
    :param iterations: The number of iterations.
    :param metric: A function that accepts two data matrices and returns their
            pair-wise distance. For a NxD and KxD matrices for instance, return
            a NxK distance matrix.
    :param center: A function that accepts a sub-matrix of X where the rows are
            points in a cluster, and returns the cluster centroid.
    :param init: A function that accepts a data matrix and k, and returns k initial centroids.
    :return: a tuple of (clustering, centroids, statistics)
    clustering - A N-dimensional vector with indices from 0 to k-1, defining the clusters.
    centroids - The kxD centroid matrix.
    """
    samples, D = X.shape
    centroids = init(X, k, metric)  # size of K x D
    clustering = np.zeros((samples))  # the i'th cell will tell us to which cluster the i'th sample belongs
    for i in range(iterations):
        # calculate the distance between the centroids and the data
        distances = metric(centroids, X)  # k, N array
        clustering = np.argmin(distances, axis=0)
        # plot_clusters(X, clustering, centroids)  - if we want to plot
        # update the centroids
        for j in range(k):
            # the j'th centroid is the center of all data points labeled j
            centroids[j, :] = center(X[np.where(clustering == j)])
    return clustering, centroids


def gaussian_kernel(X, sigma):
    """
    calculate the gaussian kernel similarity of the given distance matrix.
    :param X: A NxN distance matrix.
    :param sigma: The width of the Gaussian in the heat kernel.
    :return: NxN similarity matrix.
    """

    return np.exp(np.divide(np.square(X), -2*sigma*sigma))


def mnn(X, m):
    """
    calculate the m nearest neighbors similarity of the given distance matrix.
    :param X: A NxN distance matrix.
    :param m: The number of nearest neighbors.
    :return: NxN similarity matrix.
    """
    # get the indices of the m nearest neighbors not including the element itself
    indices = np.argsort(X, axis=1)[:, 1:m+1]
    similarity = np.zeros(X.shape)
    for i, cols in enumerate(indices):
        similarity[i, cols] = 1
    similarity = np.logical_and(similarity, similarity.T)
    return similarity


def spectral(X, k, similarity_param=0.5, similarity=gaussian_kernel,
             histogram=False):
    """
    Cluster the data into k clusters using the spectral clustering algorithm.
    :param X: A NxD data matrix.
    :param k: The number of desired clusters.
    :param similarity_param: m for mnn, sigma for the Gaussian kernel.
    :param similarity: The similarity transformation of the data.
    :return: clustering, as in the kmeans implementation.
    """
    samples, dim = X.shape
    distances = euclid(X, X)  # get the distances matrix
    if histogram:
        _ = plt.hist(np.vstack([distances, np.zeros((samples, ))]).flatten(), bins='auto')
        plt.show()
    similarity_param = np.percentile(np.vstack([distances, np.zeros((samples, ))]), 0.125)
    print("the similarity param is ", similarity_param)
    W = similarity(distances, similarity_param)
    # construct a diagonal matrix from the sum of the rows and calculate D^-1/2
    D_norm = np.diag(1/np.sqrt(np.sum(W, axis=1)))
    L_sym = np.identity(samples)-np.matmul(np.matmul(D_norm, W), D_norm)  # the symmetric laplacian
    eigvals, eigvecs = linalg.eigh(L_sym)
    # choose k lowest values
    indices = np.argsort(eigvals)
    eigvals = eigvals[indices][:k]
    eigvecs = eigvecs[:, indices][:, :k]
    # normalize eigenvecs and send to k means
    eigvecs = eigvecs / np.sqrt(np.sum(np.square(eigvecs), axis=1))[:, None]
    return kmeans(eigvecs, k, iterations=10)


def cost(X, clustering, centroids, k):
    """
    Returns the value of the cost function of k-means given the
    :param X: data of size samples x dimension
    :param clustering: array of size samples denoting to which cluster each sample belongs
    :param centroids: array of size k x dimension of centroid locations
    :param k: number of clusters
    return - the cost value
    """
    cost = 0
    for i in range(k):
        rows = np.where(clustering == i)[0]  # get the samples clustered in the i'th cluster
        cost += np.sum(euclid(X[rows, :], centroids[i, :].reshape(1,-1)))  # sum the square distances form the center
    return cost


def plot_similarity(X, k, similarity_param=0.5, similarity=gaussian_kernel,
             histogram=False):

    clustering, _ = spectral(X, k, similarity_param, similarity, histogram)
    distances = euclid(X, X)  # get the distances matrix
    W = similarity(distances, similarity_param)
    plt.imshow(W)
    plt.title("Unsorted Similarity Matrix")
    plt.show()
    indices = np.argsort(clustering)
    plt.imshow(W[indices])
    plt.title("Sorted Similarity Matrix")
    plt.show()
    return


def plot_clusters(X, clustering, centroids=-1, title=""):
    """
    Plots clusters in 2d given the data, a vector signaling in which cluster is each sample
    the centroids of each cluster and the number of clusters
    """
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=clustering)
    if type(centroids) != int:
        plt.scatter(centroids[:, 0], centroids[:, 1], s=100, marker="X")
    if len(title) > 1:
        plt.title(title)
    plt.show()
    # plt.savefig(title+".png")
    # plt.close()


def elbow():
    with open('APML_pic.pickle', 'rb') as f:
        X = pickle.load(f)
    X = circles_example()
    costs = []
    for k in range(2, 25):
        clustering, centroids = kmeans(X, k)
        title = "k="+str(k)
        # plot_clusters(X, clustering, centroids, title)
        costs.append(cost(X, clustering, centroids, k))
    plt.figure()
    plt.title("Costs as function of K - Circle Data")
    plt.plot(costs, '.')
    plt.savefig("Costs as function of K - Circle Data.png")
    plt.close()


def silhouette_method():
    sil = []
    X = circles_example()
    with open('APML_pic.pickle', 'rb') as f:
        X = pickle.load(f)
    for k in range(2, 25):
        clustering, centroids = kmeans(X, k)
        title = "k="+str(k)
        # plot_clusters(X, clustering, centroids, title)
        sil.append(silhouette(X, clustering, centroids, k))
    plt.figure()
    plt.title("Silhouette as function of K - APML Data")
    plt.plot(sil, '.')
    plt.savefig("Silhouette as function of K - APML Data.png")
    plt.close()


def silhouette(X, clustering, centroids, k):
    samples, _ = X.shape
    dists = euclid(X, X)
    sil = 0
    _, counts = np.unique(clustering, return_counts=True)
    for i in range(samples):
        icluster = clustering[i]
        cluster_size = counts[icluster] - 1
        rows = np.where(clustering == icluster)[0]
        ai = (1/cluster_size) * np.sum(dists[rows, i])
        cluster_distances = np.zeros((k,))
        for j in range(k):
            if j == icluster:
                cluster_distances[j] = np.inf  # ensuring b_i wont be the current cluster
            rows = np.where(clustering == j)[0]
            # calculate the distance between the j'th cluster's elements and x_i
            cluster_distances[j] += np.sum(dists[rows, i])
        bi = np.min(np.multiply(1/counts, cluster_distances))
        sil += (bi-ai)/max(ai, bi)
    return sil / samples


def tsne(X, dimension=2, eta=0.01, iterations=100):
    """
    X - the data set of dimension N x D
    dimension - 3 or 2, the dimension to which we reduce the data
    eta - the learning parameter for gradient descent
    iterations - number of iterations of GD
    """
    N, _ = X.shape
    X_dists = euclid(X, X)
    sigma = 1
    X_similarity = np.exp(-X_dists / (2*sigma*sigma))
    np.fill_diagonal(X_similarity, 0)  # set diagonal to 0
    Pij = (X_similarity + X_similarity.T) / (2*N)
    Y = np.random.rand((N, dimension))  # draw a random Y which will be X in low dimension
    Y_dists = euclid(Y, Y)
    Y_similarity = 1 / (1 + np.square(Y_dists))  # t-dist kernel
    Qij = Y_similarity / (Y_similarity.sum() - np.trace(Y_similarity))
    for i in range(iterations):
        Y -= eta*tsne_grad(Pij, Qij, Y, Y_similarity)
        Y_dists = euclid(Y, Y)
        Y_similarity = 1 / (1 + np.square(Y_dists))  # t-dist kernel
        Qij = Y_similarity / (Y_similarity.sum() - np.trace(Y_similarity))
    return Y


def tsne_grad(P, Q, Y, Y_similarity):
    """
    P - The original similarity matrix
    Q - normalized kernel
    Y - the lower dimension data
    Y_similarity - t-dist kernel
    """
    grad = 4 * np.multiply(
        np.multiply((P-Q).sum(axis=1), Y_similarity.sum(axis=1)),
        2*Y-Y.sum(axis=0)
    )
    return grad


def compare_tsne_pca(X):
    # load mnist dataset
    
    return


if __name__ == '__main__':
    k = 4
    with open('APML_pic.pickle', 'rb') as f:
        X = pickle.load(f)
    # X = circles_example()
    mpl.style.use('fivethirtyeight')
    # elbow()
    # clustering, centroids = kmeans(X, k)
    # plot_clusters(X, clustering, title="K Means Clustering")
    c2, cc2 = spectral(X, k)
    plot_clusters(X, c2, title="Spectral Clustering")

    # silhouette_method()
    # plot_similarity(X, k)
    # clustering = SpectralClustering(k, assign_labels="discretize", random_state=0).fit(X)
    # plot_clusters(X, clustering.labels_)
