import numpy as np
import matplotlib.pyplot as plt
import os
from concurrent.futures import ProcessPoolExecutor

import numpy.random
from tqdm import tqdm

def get_data(dat_file):
    return np.loadtxt(dat_file)

def random_sample(n, l):
    data = np.random.rand(1, l*n)
    return np.reshape(data, (l, n))

def weight(c, x_):  # this is wrong and i dont understand why
    ell = len(x_)
    # c is a constant, x is a matrix, l is len(x)
    W = np.zeros((ell, ell))
    for i_ in range(ell):
        for j in range(ell):
            W[i_][j] = np.exp(-c * (np.linalg.norm((x_[i_], x_[j]))))
    return W

def vectorised_weight(c, x):
    # W = np.exp(-c * np.square(np.mod(x[np.newaxis, :, :] - x[:, np.newaxis, :]), axis=2))
    W = np.exp(-c * (np.linalg.norm(x[np.newaxis, :, :] - x[:, np.newaxis, :], axis=2)**2))
    return W

def vectorised_diagonalD(weights):
    D = np.zeros((len(weights), len(weights)))
    np.fill_diagonal(D, np.sum(weights, axis=0))
    return D

def diagonalD(weights):
    D = np.zeros((len(weights), len(weights)))
    for i in range(len(weights)):
        for j in range(len(weights)):
            D[i][i] += weights[i][j]
    return D

def cluster(x_):  # cluster a vector
    return -1 + 2 * (x_ >= 0)

def clustering(c, x_):  # c is an unknown constant, x is the data matrix (lxl sized)
    """
    this function computes the value of c necessary to use as the hyperparamter for the weight matrix, and then
    constructs the weigth matrix, W, the diagonal matrix, D, the graph Laplacian, L, the cluster vector, v_2, as well
    as the clustering for the vector.
    :param c: value for generating value c
    :param x_: data matrix
    :return: clustering as an array
    """

    c = 2**c
    weight_matrix = vectorised_weight(c, x_)

    diagonal_matrix = vectorised_diagonalD(weight_matrix)
    laplacian = diagonal_matrix - weight_matrix
    eigenvalues, eigenvectors = np.linalg.eig(laplacian)
    v_2 = eigenvectors.T[np.argpartition(eigenvalues, 1)[1]]

    return cluster(v_2)

def c_values(limits, step):
    return np.round(np.arange(-limits, limits, step), 1)

def correct_classification():  # just used for clustering the dataset with the generated labels from the dataset
    marker = ["o", "x"]
    color = ['black', 'blue']
    dataset = get_data('twomoons.dat')
    X = dataset[:, 1:]
    for i in range(len(X)):
        temp_marker, temp_color = marker[1], color[1]
        if np.round(dataset[i][0]) == 1:
            temp_marker, temp_color = marker[0], color[0]
        plt.scatter(X[i, 0], X[i, 1], marker=temp_marker, color=temp_color)
    plt.title(f"Correct clustering")
    plt.xlabel("X_1")
    plt.ylabel("X_2")
    plt.savefig(f'./plots/clustering/cluster_baseline.png')
    plt.clf()

def original_data():  # plotting original data on its own, no clustering
    dataset = get_data('twomoons.dat')
    X = dataset[:, 1:]
    plt.scatter(X[:, 0], X[:, 1])
    plt.title(f"Original data")
    plt.xlabel("X_1")
    plt.ylabel("X_2")
    plt.savefig(f'./plots/clustering/cluster_original_data.png')
    plt.clf()

def plot_cluster(c, data, filepath):  # general function for plotting a given clustering for a c value
    marker = ["o", "x"]
    color = ['black', 'blue']
    cluster_to_plot = clustering(c, data)
    for z in range(len(cluster_to_plot)):
        temp_marker, temp_color = marker[1], color[1]
        if cluster_to_plot[z] == 1:
            temp_marker, temp_color = marker[0], color[0]
        plt.scatter(data[z, 0], data[z, 1], marker=temp_marker, color=temp_color)
    plt.title(f"Clustering for c = 2^{str(c)}")
    plt.xlabel("X_1")
    plt.ylabel("X_2")
    plt.savefig(filepath)
    plt.clf()

def calculate_cluster_error(c_value_, dataset_):  # calculating the generalisation error from a given c value and data
    data_ = dataset_[:, 1:]
    labels = dataset_[:, :1]
    cluster_ = clustering(c_value_, data_)
    mistakes_ = 0
    for z in range(len(cluster_)):
        if cluster_[z] != labels[z]:
            mistakes_ += 1
    generalisation_error_ = (mistakes_ / len(cluster_))
    return generalisation_error_, c_value_

def isotropic_gaussian(center, sd):
    cov_matirx = sd**2 * np.identity(2)
    return numpy.random.multivariate_normal(center, cov_matirx, 40)

def gaussian_clustering():
    """
    executing gaussian clustering for question 2, creating 2 sets of gaussian data from different centers, labeling
    them, combining them to form the dataset, and then running the clustering algorithm with the dataset and plotting
    the clustering along with the un-clustered data (as its randomly generated, helps with comparing)
    :return: 1 if successful
    """
    sd = 0.2
    c_array_ = c_values(20, 0.2)
    neg_class = isotropic_gaussian((-0.3, -0.3), sd)
    pos_class = isotropic_gaussian((0.15, 0.15), sd)

    labeled_neg_class = np.c_[-1 * np.ones(len(neg_class)), np.array(neg_class)]
    labeled_pos_class = np.c_[np.ones(len(pos_class)), np.array(pos_class)]
    dataset = np.vstack((labeled_neg_class, labeled_pos_class))

    generalisations = list(tuple())
    pbar = tqdm(total=len(c_array_))
    ppe = ProcessPoolExecutor(max_workers=4)
    for result in ppe.map(calculate_cluster_error, c_array_, [dataset] * len(c_array_)):
        generalisations.append(result)
        pbar.update()

    # print(min(generalisations))
    original_gaussian_cluster(neg_class, pos_class,
                              f'./plots/clustering/gaussian_cluster_original_data_{str(min(generalisations)[0])}'
                              f'_{str(min(generalisations)[1])}.png')
    plot_cluster(min(generalisations)[1], dataset[:, 1:],
                 f'./plots/clustering/gaussian_cluster_{str(min(generalisations)[0])}_{str(min(generalisations)[1])}.png')
    return 1

def original_gaussian_cluster(neg_class, pos_class, filepath):
    dataset = np.vstack((neg_class, pos_class))
    plt.scatter(dataset[:, 0], dataset[:, 1])
    plt.title(f"Original data")
    plt.xlabel("X_1")
    plt.ylabel("X_2")
    plt.savefig(filepath)
    plt.clf()

def map_labels(y_label):
    return 2 * (y_label == 1) - 1

def correct_cluster_percentage(dataset_, c_):  # implementing CP 
    data_ = dataset_[:, 1:]
    labels = dataset_[:, :1]
    mapped_labels = map_labels(labels)
    cluster_ = clustering(c_, data_)
    ell = len(data_[0])
    l_minus = 0
    for cluster_index, label in enumerate(cluster_):
        if label != mapped_labels[cluster_index]:
            l_minus += 1
    l_plus = ell - l_minus
    return max(l_plus, l_minus)/ell, c_


if __name__ == '__main__':
    if not os.path.exists('plots/clustering'):
        os.makedirs('plots/clustering')

    # marker = ["o", "x"]
    # color = ['black', 'blue']

    # c_array = c_values(20, 0.2)

    # # question 1
    # c_array = c_values(20, 0.2)
    # dataset = get_data('twomoons.dat')
    # generalisations_ = list(tuple())
    # p_bar_ = tqdm(total=len(c_array))
    # ppe = ProcessPoolExecutor(max_workers=2)
    # for result in ppe.map(calculate_cluster_error, c_array, [dataset]*len(c_array)):
    #     generalisations_.append(result)
    #     p_bar_.update()
    # # print(generalisations_)
    # print(min(generalisations_))
    #
    # plot_cluster(min(generalisations_)[1], dataset[:, 1:],
    #              f'./plots/clustering/cluster_{str(min(generalisations_)[0])}_{str(min(generalisations_)[1])}.png')

    # # plotting all values of c against the errors they produce for q1
    # for i in range(len(generalisations_)):
    #     plot_cluster(generalisations_[i][1], dataset[:, 1:],
    #                  f'./plots/clustering/cluster_{str(generalisations_[i][0])}_{str(generalisations_[i][1])}_testing.png')

    # # question 2
    # gaussian_clustering()

    # # question 3
    # dataset = get_data('dtrain123.dat')
    # dataset_13 = dataset[dataset[:, 0] != 2]
    # percentages = list(tuple())
    # cluster_values = np.round(np.arange(0, 0.11, 0.01), 2)
    # for values_index, c in enumerate(cluster_values):
    #     percentages.append(correct_cluster_percentage(dataset_13, c))
    # percentages.sort(key=lambda x: x[1])
    # print(percentages)
    # correctness_values_to_plot = [m[0] for m in percentages]
    # c_values_to_plot = [n[1] for n in percentages]
    # plt.plot(c_values_to_plot, correctness_values_to_plot)
    # plt.title(f"correctness vs c")
    # plt.xlabel("c")
    # plt.ylabel("correctness")
    # plt.savefig(f'./plots/clustering/correctness_digit_clustering_'
    #             f'{str(max(percentages)[0])}_{str(max(percentages)[1])}.png')
    # plt.clf()





