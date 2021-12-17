import numpy as np
import matplotlib.pyplot as plt
import os
import time
from concurrent.futures import ProcessPoolExecutor
# from sklearn.metrics.pairwise import euclidean_distances

import numpy.random
from tqdm import tqdm

def get_data(dat_file):
    return np.loadtxt(dat_file)

def random_sample(n, l):
    data = np.random.rand(1, l*n)
    return np.reshape(data, (l, n))

def weight(c, x_):
    l = len(x_)
    # c is a constant, x is a matrix, l is len(x)
    W = np.zeros((l, l))
    for i_ in range(l):
        for j in range(l):
            W[i_][j] = np.exp(-c * np.linalg.norm((x_[i_], x_[j])))
    return W

# def norm(x, i, j):
#     return np.sqrt(np.sum(x[i] - x[j])**2)
#     # difference = np.subtract(x[i], x[j])
#     # square_difference = np.square(difference)
#     # return square_difference
#
# def weight_2(c, x_):
#     l = len(x_)
#     # c is a constant, x is a matrix, l is len(x)
#     W = np.zeros((l, l))
#     for i_ in range(l):
#         for j in range(l):
#             W[i_][j] = np.exp(-c * np.linalg.norm((x_[i_], x_[j])))
#             # W[i_][j] = np.exp(-c * np.linalg.norm((x_[i_], x_[j])))
#     return W
#
# def weight_3(c, x_):
#     l = len(x_)
#     # c is a constant, x is a matrix, l is len(x)
#     W = np.zeros((l, l))
#     for i_ in range(l):
#         for j in range(l):
#             W[i_][j] = np.exp(-c * euclidean_distances(x_[i_], x_[j]))
#             # W[i_][j] = np.exp(-c * np.linalg.norm((x_[i_], x_[j])))
#     return W
#
# def weight_4(c, x_):
#     l = len(x_)
#     # c is a constant, x is a matrix, l is len(x)
#     W = np.zeros((l, l))
#     for i_ in range(l):
#         for j in range(l):
#             W[i_][j] = np.exp(-c * distance(x_[i_], x_[j]))
#             # W[i_][j] = np.exp(-c * np.linalg.norm((x_[i_], x_[j])))
#     return W

def distance(x, y):
    return (np.sum((x - y) ** 2)) ** 0.5

def vectorised_weight(c, x):  # seems wrong
    # print(x)
    # c is a constant, x is a matrix, l is len(x)
    # W = np.exp(-c * np.square(np.mod(x[np.newaxis, :, :] - x[:, np.newaxis, :]), axis=2))
    W = np.exp(-c * np.linalg.norm(x[np.newaxis, :, :] - x[:, np.newaxis, :], axis=2))
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

def sign(x_):
    if x_ > 0:
        return 1
    elif x_ < 0:
        return -1
    return 0

def eigenvalues_sorted(laplacian_):
    eigenValues, eigenVectors = np.linalg.eig(laplacian_)

    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]
    return eigenValues, eigenVectors

def clustering(c, x_):                                                              # c is an unknown constant, x is the data matrix (lxl sized)
    c = 2**c
    weight_matrix = weight(c, x_)                                                   # should be correct, i tested this so fucking much, its the only thing that can be wrong
    # print(weight_matrix)
    diagonal_matrix = vectorised_diagonalD(weight_matrix)                           # should be correct, tested with unvectorised and vectorised, should really be correct
    laplacian = np.subtract(diagonal_matrix, weight_matrix)                         # must be correct
    # eigenspace = np.linalg.eig(laplacian)                                         # correct
    # v_2 = (eigenspace[1][np.argsort(eigenspace[0])[1]])                           # this has to be correct but something is wrong?? it doesnt do any better than just eigenspace[1][1] but it performs worse than the line above
    eigenspace_2 = eigenvalues_sorted(laplacian)                                    # correct?
    v_2 = eigenspace_2[1][-2]                                                       # definitely should be eigenvector corresponding to the second smallest eigenvalue

    # plt.scatterplot(range(len(eigenspace[0])), eigenspace[0])
    # plt.show()
    # plt.clf()

    # if eigenspace[0][0] > eigenspace[0][1]:
    #     # print("fuck")
    #     lowest = eigenspace[0][np.argsort(eigenspace[0])[0]]
    #     second_lowest = eigenspace[0][np.argsort(eigenspace[0])[1]]
    #     if lowest > second_lowest:
    #         print("double fuck")                                        # this would print double fuck if i was wrong

    # v_2 = eigenspace[1][4]

    cluster_ = np.zeros(len(v_2))
    for z in range(len(v_2)):
        if sign(v_2[z]) == 0 or sign(v_2[z]) == 1:
            cluster_[z] = 1
        else:
            cluster_[z] = -1
        # cluster_[z] = sign(v_2[z])
        # print(f"cluster assigned was {str(cluster[z])} sign(v_2) was {str(sign(v_2[z]))}, v_2 was {str(v_2[z])}")
    return cluster_

# def vectorisedClustering(c, x):
# # not vectorised yet
#     weight_matrix = vectorised_weight(c, x)#
#     diagonal_matrix = vectorised_diagonalD(weight_matrix)
#     laplacian = diagonal_matrix - weight_matrix
#     eigenspace = np.linalg.eig(laplacian)
#     v_2 = eigenspace[1][1]
#     # print(len(v_2))
#
#     cluster = np.fromfunction(sign, v_2)
#     return cluster

def c_values(limits, step):
    return np.round(np.arange(-limits, limits, step), 1)

def correct_classification():
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

def original_data():
    dataset = get_data('twomoons.dat')
    X = dataset[:, 1:]
    plt.scatter(X[:, 0], X[:, 1])
    plt.title(f"Original data")
    plt.xlabel("X_1")
    plt.ylabel("X_2")
    plt.savefig(f'./plots/clustering/cluster_original_data.png')
    plt.clf()

def plot_cluster(c, data, filepath):
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

def calculate_cluster_error(c_value_, dataset_):
    data_ = dataset_[:, 1:]
    labels = dataset_[:, :1]
    cluster_ = clustering(c_value_, data_)
    mistakes_ = 0
    for z in range(len(cluster_)):
        if cluster_[z] != labels[z]:
            mistakes_ += 1
    generalisation_error_ = (mistakes_ / len(cluster_))
    return generalisation_error_, c_value_

def isotropic_gaussian(center, sd, covariance):
    cov_matirx = covariance * np.identity(2)
    return numpy.random.multivariate_normal(center, cov_matirx, 40)

def gaussian_clustering():
    sd = 0.2
    neg_class = isotropic_gaussian((-0.3, -0.3), sd, 0.04)
    pos_class = isotropic_gaussian((0.15, 0.15), sd, 0.01)
    # for z in range(len(neg_class)):
    #     temp_marker, temp_color = marker[0], color[0]
    #     plt.scatter(neg_class[z, 0], neg_class[z, 1], marker=temp_marker, color=temp_color)
    # for z in range(len(pos_class)):
    #     temp_marker, temp_color = marker[1], color[1]
    #     plt.scatter(pos_class[z, 0], pos_class[z, 1], marker=temp_marker, color=temp_color)
    # plt.show()
    labeled_neg_class = np.c_[np.ones(len(neg_class)), np.array(neg_class)]
    labeled_pos_class = np.c_[np.ones(len(pos_class)), np.array(pos_class)]
    dataset = np.vstack((labeled_neg_class, labeled_pos_class))
    generalisations = list(tuple())
    pbar = tqdm(total=len(c_array))
    ppe = ProcessPoolExecutor(max_workers=4)
    for result in ppe.map(calculate_cluster_error, c_array, [dataset] * len(c_array)):
        generalisations.append(result)
        pbar.update(1)
    print(generalisations)
    print(min(generalisations))
    plot_cluster(min(generalisations)[0], dataset[:, 1:],
                 f'./plots/clustering/gaussian_cluster_{str(min(generalisations)[0])}.png')

if __name__ == '__main__':
    if not os.path.exists('plots/clustering'):
        os.makedirs('plots/clustering')

    marker = ["o", "x"]
    color = ['black', 'blue']
    # dataset = get_data('twomoons.dat')
    # l = len(dataset)
    # n = len(dataset[0]-1)
    # X = dataset[:, 1:]
    # labels = dataset[:, :1]

    c_array = c_values(50, 0.2)

    dataset = get_data('twomoons.dat')
    generalisations = list(tuple())
    pbar = tqdm(total=len(c_array))
    for x in range((len(c_array))):
        cluster = clustering(c_array[x], dataset[:, 1:])
        mistakes = 0
        for i in range(len(cluster)):
            if cluster[i] != np.round(dataset[i][0]):
                mistakes += 1
        generalisation_error = (mistakes / len(cluster))
        generalisations.append((generalisation_error, c_array[x]))
        pbar.update(1)
        pbar.refresh()
    # print(generalisations)
    print(min(generalisations))

    # multiprocess uncomment bellow
    # dataset = get_data('twomoons.dat')
    # generalisations_ = list(tuple())
    # p_bar_ = tqdm(total=len(c_array))
    # ppe = ProcessPoolExecutor(max_workers=4)
    # for result in ppe.map(calculate_cluster_error, c_array, [dataset]*len(c_array)):
    #     generalisations_.append(result)
    #     p_bar_.update(1)
    # print(generalisations_)
    # print(min(generalisations_))

    # plot_cluster(min(generalisations_)[0], dataset[:, 1:], f'./plots/clustering/cluster_{str(min(generalisations_)[0])}.png')

    # gaussian_clustering()
