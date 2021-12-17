import numpy as np
import matplotlib.pyplot as plt
import os
import time
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def get_data(dat_file):
    return np.loadtxt(dat_file)

def random_sample(n, l):
    data = np.random.rand(1, l*n)
    return np.reshape(data, (l, n))

def weight(c, x_, l):
    # c is a constant, x is a matrix, l is len(x)
    W = np.zeros((l, l))
    for i_ in range(l):
        for j in range(l):
            W[i_][j] = np.exp(-c * np.linalg.norm(np.subtract(x_[i_], x_[j])))
    return W

def vectorised_weight(c, x):
    # c is a constant, x is a matrix, l is len(x)
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
    eval, evec = np.linalg.eig(laplacian_)
    eigenValues, eigenVectors = np.linalg.eig(laplacian_)

    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]
    return eigenValues, eigenVectors

def clustering(c, x_):               # c is an unknown constant, x is the data matrix (lxl sized)
    c = 2**c
    weight_matrix = vectorised_weight(c, x_)                         # should be correct
    # print(weight_matrix)
    diagonal_matrix = vectorised_diagonalD(weight_matrix)              # should be correct
    # print(diagonal_matrix[1][1])
    laplacian = diagonal_matrix - weight_matrix                         #must be correct
    eigenspace = np.linalg.eig(laplacian)                               # correct
    eigenspace_2 = eigenvalues_sorted(laplacian)
    # v_2 = (eigenspace[1][np.argsort(eigenspace[0])[1]])                 # this has to be correct but something is wrong?? it doesnt do any better than just eigenspace[1][1]
    v_2 = eigenspace_2[1][1]
    # print(np.argsort(eigenspace[0])[1])
    # print(np.round(eigenspace[0][0:5]))
    # v_3 = eigenspace[1][1]
    # print(v_2)
    # print(v_3)
    # print(eigenspace[0])
    # lowest_index = np.argmin(eigenspace[0])
    # lowest = np.partition(eigenspace[:, np.newaxis], 2)[:2]
    # print(lowest_index)
    # print(lowest)
    # print(np.argmin(lowest))
    # for i in range(len(eigenspace[0])-1):
    # print(np.argsort(eigenspace[0]))

    # if eigenspace[0][0] > eigenspace[0][1]:
    #     # print("fuck")
    #     lowest = eigenspace[0][np.argsort(eigenspace[0])[0]]
    #     second_lowest = eigenspace[0][np.argsort(eigenspace[0])[1]]
    #     if lowest > second_lowest:
    #         print("double fuck")                                        # this would print double fuck if i was wrong

    # v_2 = eigenspace[1][4]

    cluster = np.zeros(len(v_2))
    for z in range(len(v_2)):
        if sign(v_2[z]) == 0 or sign(v_2[z]) == 1:
            cluster[z] = 1
        else:
            cluster[z] = -1
        # cluster[z] = sign(v_2[z])
    return cluster

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

def correct_classification(X):
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

def original_data(X):
    plt.scatter(X[:, 0], X[:, 1])
    plt.title(f"Original data")
    plt.xlabel("X_1")
    plt.ylabel("X_2")
    plt.savefig(f'./plots/clustering/cluster_original_data.png')
    plt.clf()

def plot_cluster(c, data):
    cluster_to_plot = clustering(c, data)
    for z in range(len(cluster_to_plot)):
        temp_marker, temp_color = marker[1], color[1]
        if cluster_to_plot[z] == 1:
            temp_marker, temp_color = marker[0], color[0]
        plt.scatter(data[z, 0], data[z, 1], marker=temp_marker, color=temp_color)
    plt.title(f"Clustering for c = 2^{str(c)}")
    plt.xlabel("X_1")
    plt.ylabel("X_2")
    plt.savefig(f'./plots/clustering/cluster_{str(c)}.png')
    plt.clf()

def calculate_cluster_error(c_value_, dataset_):
    data_ = dataset_[:, 1:]
    cluster_ = clustering(c_value_, data_)
    mistakes_ = 0
    for z in range(len(cluster_)):
        if cluster_[z] != np.round(dataset_[z][0], 1):
            mistakes_ += 1
    generalisation_error_ = (mistakes_ / len(cluster_))
    return generalisation_error_, c_value_

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

    c_array = c_values(500, 0.1)

    # generalisations = list(tuple())
    # for x in range((len(c_array))):
    #     cluster = clustering(c_array[x], dataset[:, 1:])
    #     mistakes = 0
    #     for i in range(len(cluster)):
    #         if cluster[i] != np.round(dataset[i][0]):
    #             mistakes += 1
    #     generalisation_error = (mistakes / len(cluster))
    #     generalisations.append((generalisation_error, c_array[x]))
    # # print(generalisations)
    # print(min(generalisations))

    dataset = get_data('twomoons.dat')
    # generalisations_ = list(tuple())
    # p_bar_ = tqdm(smoothing=1)
    # ppe = ProcessPoolExecutor(max_workers=4)
    # for result in ppe.map(calculate_cluster_error, c_array, [dataset]*len(c_array)):
    #     generalisations_.append(result)
    #     p_bar_.update(1)

    # generalisations.sort(key=lambda x: x[1])
    # print(generalisations)
    # print(min(generalisations_))

    plot_cluster(-29.9, dataset[:, 1:])
