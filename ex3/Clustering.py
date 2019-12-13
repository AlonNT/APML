import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import pickle


def circles():
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
    circles = np.concatenate((circle1, circle2, circle3, circle4), axis=1)

    plt.figure()
    plt.plot(circles[0, :], circles[1, :], '.k')
    plt.show()

    return circles.transpose()


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
    plt.plot([-5, 5], [-5, 5], 'r')
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
    a = np.sum(np.square(X), axis=1)
    b = np.sum(np.square(Y), axis=1)
    c = np.dot(X, Y.T)
    squared_distances = a.reshape(-1, 1) + b - 2 * c

    # Due to numeric issues some values are negative but very close
    # to 0 (such as -7.6293945e-06) so we clip them to 0.
    squared_distances = np.clip(squared_distances, a_min=0, a_max=None)

    distances = np.sqrt(squared_distances)

    return distances


def euclidean_centroid(X):
    """
    return the center of mass of data points of X.
    :param X: a sub-matrix of the NxD data matrix that defines a cluster.
    :return: the centroid of the cluster.
    """
    return X.mean(axis=0)


def kmeans_pp_init(X, k, metric):
    """
    The initialization function of kmeans++, returning k centroids.
    :param X: The data matrix.
    :param k: The number of clusters.
    :param metric: a metric function like specified in the kmeans documentation.
    :return: kxD matrix with rows containing the centroids.
    """
    n, d = X.shape

    # This will holds the indices of the k centroids.
    centroids_indices = np.empty(shape=k, dtype=np.int)

    # Initialize the first centroid's index uniformly.
    centroids_indices[0] = np.random.choice(n)

    # After the first centroid was chosen, each one is sampled according to a new distribution
    # whose weights are defined by the minimal squared distances to the previous sampled centroids.
    for i in range(1, k):
        # Calculate the squared distances between each data-points and all previous sampled centroids.
        distances = np.square(metric(X, X[centroids_indices[:i]]))

        # For each data-point, calculate the minimal distance to a previous sampled centroid.
        min_distances_to_centroids = distances.min(axis=1)

        # The probability of the j-th data-point is its weight, divided by the sum of the weights.
        probabilities = min_distances_to_centroids / min_distances_to_centroids.sum()

        # Sample the next centroid according the the calculated distribution.
        centroids_indices[i] = np.random.choice(n, p=probabilities)

    return X[centroids_indices]


def kmeans(X, k, iterations=10, metric=euclid, center=euclidean_centroid, init=kmeans_pp_init):
    """
    The K-Means function, clustering the data X into k X_cluster.
    :param X: A NxD data matrix.
    :param k: The number of desired X_cluster.
    :param iterations: The number of iterations.
    :param metric: A function that accepts two data matrices and returns their
            pair-wise distance. For a NxD and KxD matrices for instance, return
            a NxK distance matrix.
    :param center: A function that accepts a sub-matrix of X where the rows are
            points in a cluster, and returns the cluster centroid.
    :param init: A function that accepts a data matrix and k, and returns k initial centroids.
    :return: a tuple of (clustering, centroids, statistics)
    clustering - A N-dimensional vector with indices from 0 to k-1, defining the X_cluster.
    centroids - The kxD centroid matrix.
    """
    # Initialize the centroids according to the given initialization function.
    centroids = init(X, k, metric)
    clustering = metric(X, centroids).argmin(axis=1)

    for iteration in range(iterations):
        # Calculate the cost of this clustering assignment.
        curr_cost = np.sum(np.square(np.min(metric(X, centroids), axis=1)))

        # Update the centroids according to the data-points in their cluster.
        for i in range(k):
            centroids[i] = center(X[clustering == i])

        # Assign each data-point to the cluster defined by the nearest centroid.
        clustering = metric(X, centroids).argmin(axis=1)

        # Calculate the cost of the clustering after updating the centroids.
        updated_cost = np.sum(np.square(np.min(metric(X, centroids), axis=1)))

        # If the cost did not decrease - the algorithm has converged.
        if not updated_cost < curr_cost:
            break

    return clustering, centroids


def gaussian_kernel(X, sigma):
    """
    calculate the gaussian kernel similarity of the given distance matrix.
    :param X: A NxN distance matrix.
    :param sigma: The width of the Gaussian in the heat kernel.
    :return: NxN similarity matrix.
    """
    return np.exp(-X ** 2 / (2 * sigma ** 2))


def mnn(X, m):
    """
    calculate the m nearest neighbors similarity of the given distance matrix.
    :param X: A NxN distance matrix.
    :param m: The number of nearest neighbors.
    :return: NxN similarity matrix.
    """
    stop = 'here'
    # TODO: YOUR CODE HERE


def spectral(X, k, similarity_param, similarity=gaussian_kernel):
    """
    Cluster the data into k clusters using the spectral clustering algorithm.
    :param X: A NxD data matrix.
    :param k: The number of desired clusters.
    :param similarity_param: m for mnn, sigma for the Gaussian kernel.
    :param similarity: The similarity transformation of the data.
    :return: clustering, as in the kmeans implementation.
    """
    n = X.shape[0]

    # Calculate the distances matrix, i.e. pairwise distance between every two data-points.
    distance_matrix = euclid(X, X)

    # Convert the distances to similarity using the given similarity function.
    similarity_matrix = similarity(distance_matrix, similarity_param)

    # Create the vector that is the diagonal of the degree (diagonal) matrix.
    diagonal_vector = np.sum(similarity_matrix, axis=1)

    # Create the matrix which is the degree matrix raised to the power of minus half.
    # Since it's a diagonal matrix, it's quicker to calculate it this way...
    degree_matrix_power_minus_half = np.diag(1 / np.sqrt(diagonal_vector))

    # Create the Laplacian matrix.
    laplacian_matrix = np.eye(n) - np.linalg.multi_dot([degree_matrix_power_minus_half,
                                                        similarity_matrix,
                                                        degree_matrix_power_minus_half])

    # Calculate the eigenvalues and the eigenvectors of the Laplacian matrix.
    # They are ordered in an ascending ordered of the eigenvalues.
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)

    # Create a matrix containing the k eigenvectors corresponding to the
    # k lowest eigenvalues, and normalize each row to be of norm 1.
    k_eigenvectors = eigenvectors[:, :k] / np.linalg.norm(eigenvectors[:, :k], axis=1).reshape(-1, 1)

    # Run the k-means algorithm on the representation using the k eigenvectors.
    clustering, centroids = kmeans(k_eigenvectors, k)

    return clustering


def main():
    mpl.style.use('seaborn')
    # N = 1000
    # X = np.empty(shape=(N, 2), dtype=np.float32)
    # X[:N // 2] = np.random.normal(loc=(3, 0), scale=2, size=(500, 2))
    # X[N // 2:] = np.random.normal(loc=(-3, 0), scale=2, size=(500, 2))
    # np.random.shuffle(X)

    X = circles()
    k = 4

    kmeans_clustering, kmeans_centroids = kmeans(X, k)

    plt.figure()
    plt.suptitle('k-means clustering')
    colors = ['.b', '.g', '.r', '.c', '.m']
    for cluster_i in range(k):
        plt.plot(X[kmeans_clustering == cluster_i, 0],
                 X[kmeans_clustering == cluster_i, 1],
                 colors[cluster_i])
    plt.plot(kmeans_centroids[:, 0], kmeans_centroids[:, 1], '*y', markersize=12)
    plt.savefig('./figures/kmeans_clustering.png')
    plt.show()

    distance_matrix = euclid(X, X)
    sigma = np.percentile(distance_matrix, q=5)

    plt.figure()
    plt.suptitle('Distances (normalized) histogram')
    plt.hist(distance_matrix.flatten(), bins=100, density=True)
    plt.savefig('./figures/distances_histogram.png')
    plt.show()

    plt.figure()
    plt.suptitle('Distances (normalized) cumulative histogram')
    plt.hist(distance_matrix.flatten(), cumulative=True, bins=100, density=True)
    plt.savefig('./figures/distances_cumulative_histogram.png')
    plt.show()

    spectral_clustering = spectral(X, k, similarity_param=0.5, similarity=gaussian_kernel)

    plt.figure()
    plt.suptitle('Spectral clustering')
    colors = ['.b', '.g', '.r', '.c', '.m']
    for cluster_i in range(k):
        plt.plot(X[spectral_clustering == cluster_i, 0],
                 X[spectral_clustering == cluster_i, 1],
                 colors[cluster_i])
    plt.savefig('./figures/spectral_clustering.png')
    plt.show()


if __name__ == '__main__':
    main()
