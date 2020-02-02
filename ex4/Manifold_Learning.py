import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


def digits_example():
    """
    Example code to show you how to load the MNIST data and plot it.
    """

    # load the MNIST data:
    digits = datasets.load_digits()
    data = digits.data / 255.
    labels = digits.target

    # plot examples:
    plt.gray()
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.axis('off')
        plt.imshow(np.reshape(data[i, :], (8, 8)))
        plt.title("Digit " + str(labels[i]))
    plt.show()


def swiss_roll_example():
    """
    Example code to show you how to load the swiss roll data and plot it.
    """

    # load the dataset:
    X, color = datasets.samples_generator.make_swiss_roll(n_samples=2000)

    # plot the data:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
    plt.show()


def faces_example(path):
    """
    Example code to show you how to load the faces data.
    """

    with open(path, 'rb') as f:
        X = pickle.load(f)

    num_images, num_pixels = np.shape(X)
    d = int(num_pixels**0.5)
    print("The number of images in the data set is " + str(num_images))
    print("The image size is " + str(d) + " by " + str(d))

    # plot some examples of faces:
    plt.gray()
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.imshow(np.reshape(X[i, :], (d, d)))
    plt.show()


def plot_with_images(X, images, title, image_num=25):
    """
    A plot function for viewing images in their embedded locations. The
    function receives the embedding (X) and the original images (images) and
    plots the images along with the embeddings.

    :param X: Nxd embedding matrix (after dimensionality reduction).
    :param images: NxD original data matrix of images.
    :param title: The title of the plot.
    :param num_to_plot: Number of images to plot along with the scatter plot.
    :return: the figure object.
    """

    n, pixels = np.shape(images)
    img_size = int(pixels**0.5)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)

    # get the size of the embedded images for plotting:
    x_size = (max(X[:, 0]) - min(X[:, 0])) * 0.08
    y_size = (max(X[:, 1]) - min(X[:, 1])) * 0.08

    # draw random images and plot them in their relevant place:
    for i in range(image_num):
        img_num = np.random.choice(n)
        x0, y0 = X[img_num, 0] - x_size / 2., X[img_num, 1] - y_size / 2.
        x1, y1 = X[img_num, 0] + x_size / 2., X[img_num, 1] + y_size / 2.
        img = images[img_num, :].reshape(img_size, img_size)
        ax.imshow(img, aspect='auto', cmap=plt.cm.gray, zorder=100000,
                  extent=(x0, x1, y0, y1))

    # draw the scatter plot of the embedded data points:
    ax.scatter(X[:, 0], X[:, 1], marker='.', alpha=0.7)

    return fig


def calc_squared_euclidean_distance_matrix(data):
    """
    Return the pair-wise euclidean distance between every pair of points in the given dataset.

    :param data: NxD matrix.
    :return: NxN euclidean distance matrix.
    """
    a = np.sum(np.square(data), axis=1)
    c = np.dot(data, data.T)
    squared_distances = a.reshape(-1, 1) + a - 2 * c

    # Due to numeric issues some values are negative but very close
    # to 0 (such as -7.6293945e-06) so we clip them to 0.
    squared_distances = np.clip(squared_distances, a_min=0, a_max=None)

    return squared_distances


def get_gaussians_2d(k=8, n=128, std=0.05):
    """
    Generate a synthetic dataset containing k gaussians,
    where each one is centered on the unit circle
    (and the distance on the sphere between each center is the same).
    Each gaussian has a standard deviation std, and contains n points.

    :param k: The amount of gaussians to create
    :param std: The standard deviation of each gaussian
    :param n: The number of points per gaussian
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


def rotate_in_high_dim_and_inject_noise(low_dim_data, dim=3, noise_std=0.125):
    """
    This function takes a data in low dimension and project it to high dimension
    while performing a random rotation and adding gaussian noise in the high dimension.

    :param low_dim_data: The low dimensional data.
    :param dim: The high dimension to use.
    :param noise_std: Standard deviation of the gaussian noise to add in the high dimension.

    :return: The high dimensional data and the rotation matrix that was used.
    """
    N, low_dim = low_dim_data.shape

    assert low_dim < dim, "The given dimension should be higher than the data's dimension!"

    # Pad each low vector with zeros to obtain a 'dim'-dimensional vectors.
    padded_data = np.pad(array=low_dim_data,
                         pad_width=((0, 0), (0, dim - low_dim)),
                         mode='constant')

    # Generate a random orthogonal matrix
    gaussian_matrix = np.random.rand(dim, dim)
    rotation_matrix, r = np.linalg.qr(gaussian_matrix)

    # Rotate the padded vectors using the random generated rotation matrix.
    rotated_2d_data_high_dim = np.dot(padded_data, rotation_matrix)

    # Add some noise to the data.
    rotated_2d_data_high_dim += np.random.normal(loc=0, scale=noise_std,
                                                 size=rotated_2d_data_high_dim.shape)

    return rotated_2d_data_high_dim, rotation_matrix


def plot_eigenvalues(eigenvalues, plot_eigenvalues_up_to, title):
    """
    :param eigenvalues: The eigenvalues to plot.
    :param plot_eigenvalues_up_to: How many eigenvalue to plot.
    :param title: Title to give the plot and the filename.
    """
    plt.figure()
    plt.suptitle(title)
    plt.xticks(np.arange(1, plot_eigenvalues_up_to + 1))
    plt.xlabel('eigenvalue index (descending order, starts from 1)')
    plt.ylabel('eigenvalue')
    plt.plot(np.arange(1, plot_eigenvalues_up_to + 1),
             eigenvalues[:plot_eigenvalues_up_to])
    plt.savefig(f'./figures/{title}.png')
    plt.show()


def scree_plot(low_dim_data=None, dim=64, n_tries=10, noise_addition=0.2):
    """
    This function takes the given low dimensional data, project it to high dimension,
    perform a random rotation, inject increasing noise (starting from 0)
    and plot the eigenvalues of the distance matrix used in the MDS algorithm.

    :param low_dim_data: The low dimensional data to work with.
                         Default is to take the 2D gaussians with its default arguments.
    :param dim: The high dimension to project to.
    :param n_tries: Number of times to inject increasing noise.
    :param noise_addition: How much to increase the std of the noise in the high dimension.
    """
    if low_dim_data is None:
        low_dim_data = get_gaussians_2d()

    N, low_dim = low_dim_data.shape
    noise_std = 0

    for k in range(n_tries):
        high_dim_data, _ = rotate_in_high_dim_and_inject_noise(low_dim_data, dim, noise_std)

        squared_euclidean_distances = calc_squared_euclidean_distance_matrix(high_dim_data)
        data_mds, eigenvalues = MDS(squared_euclidean_distances, low_dim)

        plot_eigenvalues(eigenvalues,
                         plot_eigenvalues_up_to=10,
                         title=f'MDS_distance_matrix_eigenvalues_noise_std_{noise_std:.2f}')

        noise_std += noise_addition


def mds_3d_to_2d(data_2d=None, n_tries=5, noise_addition=0.05):
    """
    This function takes some data in 2D, project it to 3D,
    perform a random rotation it and inject increasing noise (starting from 0).
    In each iteration the noise is increased and the 3D data is plotted,
    as well as the reduced 2D data from the MDS (that should be close to the original data).

    :param data_2d: The 2D data to work with.
                    Default is to take the 2D gaussians with its default arguments.
    :param n_tries: Number of times to inject increasing noise.
    :param noise_addition: How much to increase the std of the noise in 3D.
    """
    if data_2d is None:
        data_2d = get_gaussians_2d()

    noise_std = 0

    for k in range(n_tries):
        high_dim_data, rotation_matrix = rotate_in_high_dim_and_inject_noise(data_2d,
                                                                             dim=3,
                                                                             noise_std=noise_std)

        squared_euclidean_distances = calc_squared_euclidean_distance_matrix(high_dim_data)
        data_mds, eigenvalues = MDS(squared_euclidean_distances, 2)

        # Plot the data in 3D - rotated with noise.
        fig = plt.figure()
        plt.suptitle(f'Data in 3D, noise std = {noise_std:.2f}')
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(high_dim_data[:, 0], high_dim_data[:, 1], high_dim_data[:, 2])
        plt.show()

        # Plot the 2D data from the MDS output.
        fig = plt.figure()
        plt.suptitle('Data after MDS')
        ax = fig.add_subplot(111)
        ax.scatter(data_mds[:, 0],
                   data_mds[:, 1])
        plt.show()

        # Plot the eigenvalues of the distance matrix the MDS algorithm used.
        plot_eigenvalues(eigenvalues,
                         plot_eigenvalues_up_to=8,
                         title=f'MDS_distance_matrix_eigenvalues_noise_std_{noise_std:.2f}')

        # Increment the noise std for the next iteration.
        noise_std += noise_addition


def MDS(squared_euclidean_distances, d):
    """
    Given a NxN pairwise distance matrix and the number of desired dimensions,
    return the dimensionally reduced data points matrix after using MDS.

    :param squared_euclidean_distances: NxN distance matrix.
    :param d: the dimension.
    :return: Nxd reduced data point matrix.
    """
    N = squared_euclidean_distances.shape[0]

    # Compute the distance matrix induced from the given pairwise distances squared_euclidean_distances.
    H = np.eye(N) - 1 / N
    distance_matrix = (- 1/2) * H @ squared_euclidean_distances @ H

    # Diagonalize the distance matrix.
    eigenvalues, eigenvectors = np.linalg.eigh(distance_matrix)

    # NumPy returns the eigenvalues and eigenvectors in ascending order of magnitude,
    # and we want the opposite (eigenvalue in a descending order).
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    # Return the N x d matrix of columns
    # \sqrt(\lambda_i) \cdot u_i     for i = 1,...,d
    # (the eigenvalues corresponding to the largest eigenvalues
    X_reduced_dim = np.sqrt(eigenvalues[:d]) * eigenvectors[:, :d]

    return X_reduced_dim, eigenvalues


def calc_k_nearest_neighbors(data, k):
    """
    Calculate the k-nearest neighbors indices for each data-point.

    :param data: The data - NumPy array of shape (N,d) containing N samples, each with d dimensions,
    :param k: How many neighbors to extract.
    :return: A NumPy array of shape (N,k), where the i-th row contains the indices of the k-nearest neighbors
             of the i-th data-point.
    """
    # Calculate the squared euclidean pairwise distances.
    squared_distance_matrix = calc_squared_euclidean_distance_matrix(data)

    # The i-th row contains the indices of the k nearest neighbors of the i-th point.
    nearest_neighbors = np.argpartition(squared_distance_matrix, k, axis=1)[:, :k]

    return nearest_neighbors


def LLE(X, d, k):
    """
    Given a NxD data matrix, return the dimensionally reduced data matrix after
    using the LLE algorithm.

    :param X: NxD data matrix.
    :param d: the dimension.
    :param k: the number of neighbors for the weight extraction.
    :return: Nxd reduced data matrix.
    """
    N, high_dim = X.shape

    W = np.zeros(shape=(N, N))

    # We calculate the k+1 nearest neighbors, because we'll exclude the point itself so we
    # want to get k nearest neighbors which are not the point itself.
    nearest_neighbors = calc_k_nearest_neighbors(X, k + 1)

    for i in range(N):
        # Get the indices of the neighbors of the i-th data-point.
        neighbors_indices = nearest_neighbors[i]

        # Exclude the point itself from its nearest neighbors.
        neighbors_indices = np.delete(neighbors_indices, np.where(neighbors_indices == i)[0][0])

        # Get the actual data-points of the neighbors.
        neighbors = X[neighbors_indices]

        # Create the z's vectors, which are the subtraction of the current i-th data-point
        # from the neighbors vectors.
        neighbors -= X[i]

        # Calculate the Gram matrix of the neighbors: the ij-th coordinate is the
        # inner-product between the i-th data-point and the j-th data-point.
        gram_matrix = np.dot(neighbors, neighbors.T)

        # Calculate the pseudo-inverse of the Gram matrix.
        gram_matrix_inv = np.linalg.pinv(gram_matrix)

        # Calculate the weight vector of this data-point, as the multiplication of the
        # inverse Gram matrix with the ones vector.
        w_unnormalized = gram_matrix_inv @ np.ones(k)

        # Normalize w to make it sum to 1.
        w = w_unnormalized / w_unnormalized.sum()

        W[i, neighbors_indices] = w

    # Calculate the matrix M, defined by the identity matrix minus W.
    M = np.eye(N) - W

    # Decompose M^T * M into its EVD, where the eigenvalues are sorted in an ascending order.
    eigenvalues, eigenvectors = np.linalg.eigh(M.T @ M)

    # Return the d eigenvectors corresponding to the lowest eigenvalue
    # (excluding the first eigenvalue which is 0).
    return eigenvectors[:, 1: d+1]


def DiffusionMap(data, d, sigma, t):
    """
    Given a NxD data matrix, return the dimensionally reduced data matrix after
    using the Diffusion Map algorithm. The k parameter allows restricting the
    kernel matrix to only the k nearest neighbor of each data point.

    :param data: NxD data matrix.
    :param d: the dimension.
    :param sigma: the sigma of the gaussian for the kernel matrix transformation.
    :param t: the scale of the diffusion (amount of time steps).
    :return: Nxd reduced data matrix.
    """
    # Calculate the similarity matrix of the data-points.
    similarity_matrix = get_similarity_matrix(data, sigma_as_percentile=sigma)

    # Normalize each row to sum to 1, to create a Markov Transition Matrix.
    similarity_matrix /= similarity_matrix.sum(axis=1).reshape(-1, 1)

    # Diagonalize the similarity matrix.
    eigenvalues, eigenvectors = np.linalg.eigh(similarity_matrix)

    # NumPy returns the eigenvalues and eigenvectors in ascending order of magnitude,
    # and we want the opposite (eigenvalue in a descending order).
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    # Select only the eigenvectors corresponding to the 2...(d + 1) highest eigenvalues.
    d_eigenvectors = eigenvectors[:, 1:d + 1]
    d_eigenvalues = eigenvalues[1:d + 1]

    # Return those eigenvectors, where the i-th eigenvector is multiplied by (\lambda_i)^t
    return np.power(d_eigenvalues, t) * d_eigenvectors


def mds_2d_moons():
    """
    Project the moons dataset in 2D to 3D while random rotating and injecting noise,
    and show the results of the MDS dimensionality reduction.
    """
    moons_points, moons_labels = datasets.samples_generator.make_moons(n_samples=256)

    # Plot the generated points.
    plt.figure()
    plt.scatter(moons_points[:, 0], moons_points[:, 1], s=5)
    plt.show()

    mds_3d_to_2d(data_2d=moons_points)


def swiss_roll_playground():
    """
    Perform a dimensionality reduction of the swiss roll dataset in 3D to 2D,
    using different dimensionality-reduction algorithms and comparing the results.
    """
    swiss_roll, swiss_roll_color = datasets.samples_generator.make_swiss_roll(n_samples=2000)

    swiss_roll_2d_lle = LLE(swiss_roll, d=2, k=11)
    swiss_roll_2d_mds, _ = MDS(calc_squared_euclidean_distance_matrix(swiss_roll), d=2)

    # Plot the original data in 3D:
    fig = plt.figure()
    title = 'Swiss-Roll_original_3d_data'
    plt.suptitle(title)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(swiss_roll[:, 0], swiss_roll[:, 1], swiss_roll[:, 2],
               c=swiss_roll_color, cmap=plt.cm.Spectral)
    plt.savefig(f'./figures/{title}.png')

    for swiss_roll_2d, dim_reduction_type in ((swiss_roll_2d_mds, 'MDS'),
                                              (swiss_roll_2d_lle, 'LLE')):
        plt.figure()
        title = f'Swiss-Roll_2d_using_{dim_reduction_type}'
        plt.suptitle(title)
        plt.scatter(swiss_roll_2d[:, 0], swiss_roll_2d[:, 1],
                    c=swiss_roll_color, cmap=plt.cm.Spectral)
        plt.savefig(f'./figures/{title}.png')

    plt.show()


def get_similarity_matrix(data, sigma_as_percentile):
    """
    Calculate the similarity matrix for the given data.

    :param data: A (N, d) dimensional array of data to operate on.
    :param sigma_as_percentile: which sigma to use, given as a percentile
                                of the pairwise distances.
    :return: A (N, N) pairwise similarity matrix.
    """
    # Calculate the pairwise distance matrix to take extract percentile out of it.
    distance_matrix = calc_squared_euclidean_distance_matrix(data)

    # Take the upper triangular of the distances matrix, excluding the main diagonal.
    # This is because the matrix is symmetric, and the main diagonal is the distance of each
    # data-point to itself, which is zero.
    distances = distance_matrix[np.triu_indices_from(distance_matrix, k=1)].flatten()
    sigma = np.percentile(distances, q=sigma_as_percentile)

    # If the q is really small, the corresponding percentile could be 0.
    # This is an invalid value for sigma, so we continue to the next iteration.
    if sigma == 0:
        raise ValueError(f"The {sigma_as_percentile:.2f} percentile of the distances is zero!")

    similarity_matrix = np.exp(-distance_matrix ** 2 / (2 * sigma ** 2))

    return similarity_matrix


def diffusion_map_swiss_roll(sigmas=None, ts=None):
    """
    Try different parameters of sigma and t to use in the DiffusionMap dimensionality
    reduction algorithm.

    :param sigmas: an iterable of sigma values to try.
    :param ts: an iterable of t values to try.
    """
    data, color = datasets.samples_generator.make_swiss_roll(n_samples=2000)

    if sigmas is None:
        sigmas = np.arange(1, 3, 0.5)

    if ts is None:
        ts = np.arange(1, 52, 10)

    fig = plt.figure()
    plt.suptitle("Swiss_Roll_original_3d_data")
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=color, cmap=plt.cm.Spectral)
    plt.savefig(f'./figures/Swiss_Roll_original_3d_data.png')
    plt.show()

    for sigma in sigmas:
        for t in ts:
            reduced_dim_data = DiffusionMap(data, 2, sigma, t)

            plt.figure()
            plt.suptitle(f'Swiss_Roll_2d_using_DM_sigma_{sigma}_t_{t}')
            plt.scatter(reduced_dim_data[:, 0], reduced_dim_data[:, 1], c=color, cmap=plt.cm.Spectral)
            plt.savefig(f'./figures/Swiss_Roll_2d_using_DM_sigma_{sigma}_t_{t}.png')

    plt.show()


def plot_digits_embedding(digits_embedded, targets, targets_names, title):
    """
    Plot the given embedding of the digits dataset in 2D.

    :param digits_embedded: The embedding of the digits to plot.
    :param targets: label for each data-point.
    :param targets_names: a list containing the labels names.
    :param title: title of the plot, used in the file-name and in the header.
    """
    plt.figure()
    plt.suptitle(title)
    color = iter(plt.cm.rainbow(np.linspace(0, 1, 10)))
    for digit in targets_names:
        plt.scatter(digits_embedded[targets == digit, 0],
                    digits_embedded[targets == digit, 1],
                    s=5, c=next(color).reshape(1, -1),
                    label=digit)
    plt.legend()
    plt.savefig(f'./figures/{title}.png')
    plt.show()


def mnist_embeddings():
    """
    Try different dimensionality reduction algorithms on the digits dataset.
    """
    dataset = datasets.load_digits()
    digits = dataset.data
    targets = dataset.target
    targets_names = dataset.target_names

    digits_pca = PCA(n_components=2).fit_transform(digits)
    plot_digits_embedding(digits_pca, targets, targets_names, title='digits_PCA')

    digits_mds, _ = MDS(calc_squared_euclidean_distance_matrix(digits), d=2)
    plot_digits_embedding(digits_mds, targets, targets_names, title='digits_MDS')

    digits_lle_by_param = dict()
    for k in [2, 3, 5, 7, 11, 13, 17, 19, 23]:
        digits_lle_by_param[k] = LLE(digits, 2, k)
        plot_digits_embedding(digits_lle_by_param[k], targets, targets_names,
                              title=f'digits_LLE_{k}_nearest_neighbors')

    digits_dm_by_params = dict()
    for sigma in [1, 2, 3, 4]:
        for t in [3, 5, 7, 11]:
            digits_dm_by_params[(sigma, t)] = DiffusionMap(digits, 2, sigma, t)
            plot_digits_embedding(digits_dm_by_params[(sigma, t)], targets, targets_names,
                                  title=f'digits_DiffusionMap_sigma_{sigma}_t_{t}')

    digits_tsne = TSNE(n_components=2).fit_transform(digits)
    plot_digits_embedding(digits_tsne, targets, targets_names, title='digits_tSNE')

    plt.show()


def faces_embeddings(path='./faces.pickle'):
    """
    Try different dimensionality reduction algorithms on the faces dataset.
    """
    with open(path, 'rb') as f:
        faces = pickle.load(f)

    title = 'faces_PCA'
    faces_pca = PCA(n_components=2).fit_transform(faces)
    plot_with_images(faces_pca, faces, title)
    plt.savefig(f'./figures/{title}.png')
    plt.show()

    title = 'faces_MDS'
    faces_mds, _ = MDS(calc_squared_euclidean_distance_matrix(faces), d=2)
    plot_with_images(faces_mds, faces, title)
    plt.savefig(f'./figures/{title}.png')
    plt.show()

    faces_lle_by_param = dict()
    for k in [3, 5, 7, 11, 13, 17, 19, 23]:
        title = f'faces_LLE_{k}_nearest_neighbors'
        faces_lle_by_param[k] = LLE(faces, 2, k)
        plot_with_images(faces_lle_by_param[k], faces, title)
        plt.savefig(f'./figures/{title}.png')
        plt.show()

    faces_dm_by_params = dict()
    for sigma in [1, 2, 3, 4, 5, 6]:
        for t in [3, 5, 7, 11, 17]:
            title = f'faces_DiffusionMap_sigma_{sigma}_t_{t}'
            faces_dm_by_params[(sigma, t)] = DiffusionMap(faces, 2, sigma, t)
            plot_with_images(faces_dm_by_params[(sigma, t)], faces, title)
            plt.savefig(f'./figures/{title}.png')
            plt.show()

    title = 'faces_tSNE'
    faces_tsne = TSNE(n_components=2).fit_transform(faces)
    plot_with_images(faces_tsne, faces, title)
    plt.savefig(f'./figures/{title}.png')
    plt.show()


def main():
    # scree_plot()
    # mds_2d_moons()
    # mds_3d_to_2d(data_2d=get_gaussians_2d())
    # swiss_roll_example()
    # swiss_roll_playground()
    # diffusion_map_swiss_roll()
    # mnist_embeddings()
    faces_embeddings()


if __name__ == '__main__':
    main()
