import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import pickle


def get_circles():
    """
    an example function for generating and plotting synthetic data.
    :returns: An array of shape (N, 2) where each row contains a 2D point in the dataset.
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
    plt.scatter(circles[0, :], circles[1, :], s=5)
    plt.show()

    return circles.transpose()


def get_apml_pic(path='./Clustering_Code/APML_pic.pickle'):
    """
    an example function for loading and plotting the APML_pic data.

    :param path: the path to the APML_pic.pickle file
    :returns: An array of shape (N, 2) where each row contains a 2D point in the dataset.
    """

    with open(path, 'rb') as f:
        apml = pickle.load(f)

    plt.scatter(apml[:, 0], apml[:, 1], s=5)
    plt.show()

    return apml


def microarray_exploration(data_path='./Clustering_Code/microarray_data.pickle',
                           genes_path='./Clustering_Code/microarray_genes.pickle',
                           conds_path='./Clustering_Code/microarray_conds.pickle'):
    """
    an example function for loading and plotting the microarray data.
    Specific explanations of the data set are written in comments inside the
    function.

    :param data_path: path to the data file.
    :param genes_path: path to the genes file.
    :param conds_path: path to the conds file.
    :returns: An array of shape (N_genes, N_conds) where each row
              contains the responses of the specific gene to the conditions.
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

    return data, genes, conds


def get_gaussians(k=7, n=100, std=0.17):
    """
    Generate a synthetic dataset containing k gaussians,
    where each one is centered on the unit circle
    (and the distance on the sphere between each center is the same).
    Each gaussian has a standard deviation std, and contains n points.
    :returns: An array of shape (N, 2) where each row contains a 2D point in the dataset.
    """
    # Generate the angles of each on of the k centers.
    angles = np.linspace(start=0, stop=2 * np.pi, num=k, endpoint=False)

    # Generate the points themselves by taking sin and cos, on the unit ball.
    centers = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    # Create an empty array that will contain the generated points.
    points = np.empty(shape=(k * n, 2), dtype=np.float64)

    # For each one of the k centers, generate the points by
    # sampling from a normal distribution in each axis.
    for i in range(k):
        points[i * n: i * n + n, 0] = np.random.normal(loc=centers[i, 0], scale=std, size=n)
        points[i * n: i * n + n, 1] = np.random.normal(loc=centers[i, 1], scale=std, size=n)

    # Plot the generated points.
    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], s=5)
    plt.show()

    return points


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
        if np.isnan(probabilities).any():
            stop = 'here'
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
    # Initialize the centroids according to the given initialization function,
    # and assign the clusters accordingly.
    centroids = init(X, k, metric)
    clustering = metric(X, centroids).argmin(axis=1)

    for iteration in range(iterations):
        # Copy the previous centroids to compare with the new ones.
        previous_centroids = np.copy(centroids)

        # Assign each data-point to the cluster defined by the nearest centroid.
        clustering = metric(X, centroids).argmin(axis=1)

        # Update the centroids according to the data-points in their cluster.
        for i in range(k):
            centroids[i] = center(X[clustering == i])

        # If the centroids did not change - the algorithm converged.
        if np.allclose(previous_centroids, centroids):
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
    # The ij-th entry will be True iff the j-th data-point is
    # in the m nearest-neighbors of the i-th data-point.
    j_is_neighbor_to_i = np.zeros_like(X, dtype=np.bool)

    # The ij-th entry will be True iff the i-th data-point is
    # in the m nearest-neighbors of the j-th data-point.
    i_is_neighbor_to_j = np.zeros_like(X, dtype=np.bool)

    # The i-th row contains the indices of the m nearest neighbors of the i-th point.
    nearest_neighbors = np.argpartition(X, m, axis=1)[:, :m]

    # Fill each one of the two arrays accordingly.
    j_is_neighbor_to_i[np.repeat(np.arange(X.shape[0]), m),
                       nearest_neighbors.flatten()] = True
    i_is_neighbor_to_j[nearest_neighbors.flatten(),
                       np.repeat(np.arange(X.shape[0]), m)] = True

    # Return the OR of the two boolean arrays defined above.
    return np.logical_or(j_is_neighbor_to_i, i_is_neighbor_to_j).astype(np.float)


def plot_eigenvalues(eigenvalues, plot_eigenvalues_up_to, title):
    plt.figure()
    plt.suptitle(title)
    plt.xticks(np.arange(1, plot_eigenvalues_up_to + 1))
    plt.xlabel('eigenvalue index (ascending order, starts from 1)')
    plt.ylabel('eigenvalue')
    plt.plot(np.arange(1, plot_eigenvalues_up_to + 1),
             eigenvalues[:plot_eigenvalues_up_to])
    plt.savefig(f'./figures/{title}.png')
    plt.show()


def spectral(X, k, similarity_param, similarity=gaussian_kernel,
             plot_eigenvalues_up_to=None):
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

    if plot_eigenvalues_up_to is not None:
        plot_eigenvalues(eigenvalues, plot_eigenvalues_up_to,
                         title=f'{similarity.__name__}_{similarity_param}_eigenvalues')

    # These are the norms of each row in the eigenvectors,
    # needed in order to normalize each row to be a unit-vector.
    row_norms = np.linalg.norm(eigenvectors[:, :k], axis=1).reshape(-1, 1)
    non_zero_rows = (row_norms.flatten() != 0)

    # Create a matrix containing the k eigenvectors corresponding to the
    # k lowest eigenvalues, and normalize each row to be of norm 1.
    # Rows with 0 norm (meaning all row is zero) can't be normalized
    # to a unit-vector, so these vectors remains 0-vector.
    k_eigenvectors = eigenvectors[:, :k]
    k_eigenvectors[non_zero_rows] /= row_norms[non_zero_rows]

    # Run the k-means algorithm on the representation using the k eigenvectors.
    clustering, centroids = kmeans(k_eigenvectors, k)

    return clustering


def plot_clustering(points, k, clustering, title, centroids=None):
    """
    Plot the clustering of the data-points.
    Also saves the figure in the directory 'figures'.
    :param points: The data-points to plot.
                   Should be 2-dimensional, since they will be plotted in R^2.
    :param k: Amount of clusters.
    :param clustering: An array indicating for each data-point the index of the assigen cluster.
    :param title: The title of the plot - will be used both as a title to the figure
                  and in the filename of the saved figure.
    :param centroids: If given - plot the centroids as well.
    """
    plt.figure()
    plt.suptitle(title)
    color = iter(plt.cm.rainbow(np.linspace(0, 1, k)))
    for cluster_i in range(k):
        plt.scatter(points[clustering == cluster_i, 0],
                    points[clustering == cluster_i, 1],
                    s=5, c=next(color).reshape(1, -1))
    if centroids is not None:
        plt.plot(centroids[:, 0], centroids[:, 1], '*y', markersize=12)
    plt.savefig(f'./figures/{title}.png')
    plt.show()


def plot_distance_matrix_histograms(distance_matrix, data_name):
    """
    Plot the distance matrix histograms - both regular histogram and cumulative.
    :param distance_matrix: The distance matrix to plot.
    :param name: Name of the data the distances were taken from.
    """
    # Take the upper triangular of the distances matrix, excluding the main diagonal.
    # This is because the matrix is symmetric, and the main diagonal is the distance of each
    # data-point to itself, which is zero.
    distances = distance_matrix[np.triu_indices_from(distance_matrix, k=1)].flatten()

    # Plot the histogram.
    plt.figure()
    plt.suptitle(f'{data_name} distances histogram')
    plt.hist(distances, bins=100, density=True)
    plt.savefig(f'./figures/{data_name}_distances_histogram.png')
    plt.show()

    # Plot the cumulative histogram.
    plt.figure()
    plt.suptitle(f'{data_name} distances cumulative histogram')
    plt.hist(distances, cumulative=True, bins=100, density=True)
    plt.savefig(f'./figures/{data_name}_distances_cumulative_histogram.png')
    plt.show()


def plot_similarity_matrices(X, spectral_clustering,
                             similarity, similarity_param,
                             data_name, kernel_name, param_name):
    """
    Plot the similarity matrices - based on the distances taken from the shuffled data points,
    and based on the distances taken from the data points after sorting according to the clusters.
    :param X: The data points
    :param spectral_clustering: The clustering of the data-points.
    :param similarity: A function that creates a similarity matrix from a distances matrix.
    :param similarity_param: The parameter of the similarity (sigma in a gaussian kernel, and m in
                             m-nearest neighbors kernel).
    :param data_name: The name of the data the data-points are from.
    :param kernel_name: The name of the kernel (gaussian / mnn).
    :param param_name: The parameter name.
    """
    # Go through the shuffled data-points, and the data-points ordered according to the clusters.
    for data, sort_type in [(np.random.permutation(X), 'shuffled_data'),
                            (X[np.argsort(spectral_clustering)], 'sorted_data')]:
        # Calculate the distance matrix and the similarity matrix.
        distance_matrix = euclid(data, data)
        similarity_matrix = similarity(distance_matrix, similarity_param)

        # Plot the similarity matrix, treating it as a grayscale image.
        plt.figure()
        title = f'{data_name}_similarity_matrix_{sort_type}_{kernel_name}_{param_name}'
        plt.suptitle(title)
        plt.imshow(similarity_matrix, cmap='gray')
        plt.savefig(f'./figures/{title}.png')
        plt.show()


def run_clustering():
    """
    The main function which runs the clustering algorithms (k-means and spectral clustering)
    on the two datasets (APML and circles).
    """
    for data_name, X, k in [
        # ('APML', get_apml_pic(), 9),
        # ('circles', get_circles(), 4),
        ('5_gaussians', get_gaussians(k=5, n=100, std=0.25), 5),
        ('11_gaussians', get_gaussians(k=11, n=100, std=0.11), 11),
    ]:
        # Cluster using k-means.
        kmeans_clustering, kmeans_centroids = kmeans(X, k)

        # Plot the clusters.
        plot_clustering(X, k, kmeans_clustering,
                        title=f'{data_name}_kmeans_clustering', centroids=kmeans_centroids)

        # Calculate the distance matrix and plot the histogram of the values.
        distance_matrix = euclid(X, X)
        plot_distance_matrix_histograms(distance_matrix, data_name)

        # Try many different similarity params,
        # both for the gaussian kernel and for the m-nearest neighbors.
        for similarity_param, similarity in [
            (0.0625, gaussian_kernel),
            (0.125, gaussian_kernel),
            (0.25, gaussian_kernel),
            (0.5, gaussian_kernel),
            (1, gaussian_kernel),
            (3, mnn),
            (5, mnn),
            (7, mnn),
            (9, mnn),
            (11, mnn),
            (13, mnn),
            (15, mnn),
            (17, mnn),
            (19, mnn),
            (21, mnn),
        ]:
            if similarity == gaussian_kernel:
                kernel_name = 'gaussian'
                param_name = f'{similarity_param}_percentile'

                # Take the upper triangular of the distances matrix, excluding the main diagonal.
                # This is because the matrix is symmetric, and the main diagonal is the distance of each
                # data-point to itself, which is zero.
                distances = distance_matrix[np.triu_indices_from(distance_matrix, k=1)].flatten()
                similarity_param = np.percentile(distances, q=similarity_param)

                # If the q is really small, the corresponding percentile could be 0.
                # This is an invalid value for sigma, so we continue to the next iteration.
                if similarity_param == 0:
                    continue
            else:
                kernel_name = 'mnn'
                param_name = str(similarity_param)

            spectral_clustering = spectral(X, k, similarity_param, similarity)

            plot_clustering(X, k, spectral_clustering,
                            title=f'{data_name}_spectral_clustering_{kernel_name}_{param_name}')

            plot_similarity_matrices(X, spectral_clustering,
                                     similarity, similarity_param,
                                     data_name, kernel_name, param_name)


def elbow(data, max_k, n_tries=10, plot_clusters=False, data_name=''):
    """
    Run k-means on the data-points, trying many different option for k (from 1 to max_k, inclusive).
    Each k will be tried n_tries, and run that achieved the lowest cost will be chosen.
    Then, plot the cost as a function of k, allowing to see the 'elbow' and pick the best k.
    :param data: The data to cluster.
    :param max_k: Maximal k to try. The k's will be 1,2,...,max_k.
    :param n_tries: Number of tries per k.
    :param plot_clusters: Whether or not to plot the clustering for each k.
    :param data_name: The name of the dataset.
    """
    # This will hold the cost per k.
    costs = np.zeros(shape=max_k)

    for k in range(1, max_k + 1):
        best_cost, best_clustering, best_centroids = np.inf, None, None
        for _ in range(n_tries):
            clustering, centroids = kmeans(data, k)
            curr_cost = np.sum(np.square(np.min(euclid(data, centroids), axis=1)))
            if curr_cost < best_cost:
                best_cost = curr_cost
                best_clustering = np.copy(clustering)
                best_centroids = np.copy(centroids)

        costs[k-1] = best_cost

        if plot_clusters:
            plot_clustering(data, k, best_clustering,
                            title=f'{data_name}_{k}_clusters', centroids=best_centroids)

    # Plot the cost as a function of k, showing the 'elbow' effect.
    plt.figure()
    title = f'{data_name}_costs_graph'
    plt.suptitle(title)
    plt.xticks(np.arange(1, max_k + 1))
    plt.xlabel('k')
    plt.ylabel('cost')
    plt.plot(np.arange(1, max_k + 1), costs)
    plt.savefig(f'./figures/{title}.png')
    plt.show()


def silhouette(data, clustering_function, max_k,
               n_tries=10, plot_clusters=False, data_name='',
               clustering_kwargs=None):
    """
    Try different k's and plot the silhouette scores in order to choose the best k.
    :param data: The data to cluster.
    :param clustering_function: The clustering function.
                                Should be kmeans or spectral.
    :param max_k: Maximal k to try. The k's will be 2,3,...,max_k.
    :param n_tries: Number of tries per k.
    :param plot_clusters: Whether or not to plot the clustering for each k.
    :param data_name: The name of the dataset.
    :param clustering_kwargs: Additional arguments for the clustering function
                              (needed for the spectral clustering).
    """
    if clustering_kwargs is None:
        clustering_kwargs = dict()

    # In spectral clustering the similarity param is a percentile from all
    # the pairwise distances, so set it accordingly.
    if clustering_function == spectral and clustering_kwargs['similarity'] == gaussian_kernel:
        # Take the upper triangular of the distances matrix, excluding the main diagonal.
        # This is because the matrix is symmetric, and the main diagonal is the distance of each
        # data-point to itself, which is zero.
        distance_matrix = euclid(data, data)
        distances = distance_matrix[np.triu_indices_from(distance_matrix, k=1)].flatten()
        similarity_param = np.percentile(distances, q=clustering_kwargs['similarity_param'])
        clustering_kwargs['similarity_param'] = similarity_param

        # If the q is really small, the corresponding percentile could be 0.
        # This is an invalid value for sigma, so we continue to the next iteration.
        if similarity_param == 0:
            print('sigma for the gaussian kernel is zero, can not work with that...')
            return

    n_points = data.shape[0]

    # This will hold the silhouette score per k = 2,3,...,max_k
    silhouette_scores = np.zeros(shape=max_k - 1)

    for k in range(2, max_k + 1):
        best_clustering, best_silhouette_score = None, -np.inf
        for _ in range(n_tries):
            clustering = clustering_function(data, k, **clustering_kwargs)

            # In the k-means clustering function, two values are returned -
            # clustering & centroids.
            # Since we don't need the centroids here we just take the clustering.
            if clustering_function == kmeans:
                clustering = clustering[0]

            # Now, calculate the silhouette score of this clustering.

            # a_i will give us a measure of how well example i is assigned to
            # its own cluster. The smaller a_i is, the better.
            # a_i is the average distance between x_i and the other points
            # within its own cluster.
            a = np.zeros(shape=n_points, dtype=np.float)

            # b_i gives us a similar measure of how close it is to the closest
            # other cluster. The largest b_i is, the better.
            # b_i is the average distance between x_i and the other points
            # within the closest cluster to x_i which isn't its own cluster.
            b = np.zeros(shape=n_points, dtype=np.float)

            # Go through each sample in the data and calculate its a and b scores.
            for i in range(n_points):
                # Calculate the average distance between the i-th data point
                # and the other points in each of the clusters.
                d = np.zeros(shape=k, dtype=np.float)
                for j in range(k):
                    cluster_points_mask = (clustering == j)
                    not_i_mask = np.ones_like(cluster_points_mask)
                    not_i_mask[i] = False
                    in_cluster_other_points_mask = cluster_points_mask & not_i_mask

                    # If there are not points in the cluster which are not the i-th point
                    # the mean distance to these points is defined as zero.
                    if not np.any(in_cluster_other_points_mask):
                        continue

                    d[j] = np.mean(euclid(data[i].reshape(1, -1),
                                          data[in_cluster_other_points_mask]))

                # Calculate the average distance between the i-th data point
                # and the other points within its own cluster.
                a[i] = d[clustering[i]]

                # Calculate the average distance between the i-th data point
                # and the other points within the closest cluster
                # to that data point which isn't its own cluster.
                not_ith_cluster_mask = np.ones(k, dtype=np.bool)
                not_ith_cluster_mask[clustering[i]] = False
                b[i] = np.min(d[not_ith_cluster_mask])

            assert np.all(np.max([a, b], axis=0) > 0)

            silhouette_score = np.mean((b - a) / np.max([a, b], axis=0))

            if silhouette_score > best_silhouette_score:
                best_silhouette_score = silhouette_score
                best_clustering = clustering

        silhouette_scores[k-2] = best_silhouette_score

        if plot_clusters:
            plot_clustering(data, k, best_clustering, title=f'{data_name}_{k}_clusters')

    # Plot the silhouette score as a function of k, showing the 'elbow' effect.
    plt.figure()
    title = f'{data_name}_{clustering_function.__name__}'
    if clustering_function == spectral:
        title += f'_{clustering_kwargs["similarity"].__name__}'
        title += f'_{clustering_kwargs["similarity_param"]}'
    title += '_silhouettes'
    plt.suptitle(title)
    plt.xlabel('k')
    plt.ylabel('silhouette score')
    plt.bar(np.arange(2, max_k + 1), silhouette_scores, tick_label=np.arange(2, max_k + 1))
    plt.savefig(f'./figures/{title}.png')
    plt.show()


def cluster_biological_data(max_k=15):
    data, genes, conds = microarray_exploration()
    elbow(data, max_k, n_tries=10, plot_clusters=False, data_name='biological')

    # Try many different similarity params,
    # both for the gaussian kernel and for the m-nearest neighbors.
    for similarity_param, similarity in [
        (1, gaussian_kernel),
        (2, gaussian_kernel),
        (4, gaussian_kernel),
        (8, gaussian_kernel),
        (7, mnn),
        (11, mnn),
        (17, mnn),
        (23, mnn),
    ]:
        # Try to cluster using spectral clustering with the above similarity kernel
        # and similarity_param, using k = 2,3,...,max_k and plot the silhouette scores
        silhouette(data, spectral, max_k,
                   n_tries=10, plot_clusters=False, data_name='biological',
                   clustering_kwargs={'similarity': similarity,
                                      'similarity_param': similarity_param})

        # Plot the eigenvalues of the spectral clustering using the above
        # similarity kernel and similarity_param.
        spectral(data, max_k,
                 similarity_param=similarity_param, similarity=similarity,
                 plot_eigenvalues_up_to=max_k)


def main():
    # run_clustering()

    # n_gaussians = 5
    # max_k = 15
    # data = get_gaussians(k=n_gaussians, n=100, std=0.25)
    #
    # elbow(data, max_k, n_tries=10, plot_clusters=True, data_name=f'{n_gaussians}_gaussians')
    #
    # n_gaussians = 11
    # max_k = 20
    # data = get_gaussians(k=n_gaussians, n=100, std=0.11)
    #
    # clustering = spectral(data, k=n_gaussians,
    #                       similarity_param=0.1, similarity=gaussian_kernel,
    #                       plot_eigenvalues_up_to=max_k)
    # plot_clustering(data, n_gaussians, clustering,
    #                 title=f'{n_gaussians}_gaussians_spectral_clustering')
    #
    # silhouette(data, spectral, max_k,
    #            n_tries=10, plot_clusters=True, data_name=f'{n_gaussians}_gaussians',
    #            clustering_kwargs={'similarity': mnn,
    #                               'similarity_param': 5})

    cluster_biological_data()


if __name__ == '__main__':
    mpl.style.use('seaborn')
    main()
