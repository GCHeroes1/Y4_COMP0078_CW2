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


def exponential(c, x, i, j):
    return np.exp(-c * np.linalg.norm(x[i] - x[j]))


def exponential_2(xi, xj):
    return np.exp(np.linalg.norm(xi - xj))


def weight(c, x, l):
    # c is a constant, x is a matrix, l is len(x)
    W = np.zeros((l, l))
    for i in range(l):
        for j in range(l):
            W[i][j] = np.exp(-c * np.linalg.norm(np.subtract(x[i], x[j])))
    return W

def vectorised_weight(c, x):
    # c is a constant, x is a matrix, l is len(x)
    W = np.exp(-c * np.linalg.norm(x[np.newaxis, :, :] - x[:, np.newaxis, :], axis=2))
    return W


def test(x, y):
    return x ** y


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


def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0


def clustering(c, x):
    c = 2**c
    weight_matrix = vectorised_weight(c, x)                         # should be correct
    # print(weight_matrix)
    diagonal_matrix = vectorised_diagonalD(weight_matrix)              # should be correct
    # print(diagonal_matrix[1][1])
    laplacian = diagonal_matrix - weight_matrix                         #must be correct
    eigenspace = np.linalg.eig(laplacian)                               # correct
    v_2 = (eigenspace[1][np.argsort(eigenspace[0])[1]])                 # this has to be correct but something is wrong?? it doesnt do any better than just eigenspace[1][1]
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
        print(dataset[i][0])
        # print(np.round(X[i][0]))
        if np.round(dataset[i][0]) == 1:
            # print("it was 1")
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

def calculate_cluster_error(c_value, dataset_):
    data = dataset_[:, 1:]
    cluster = clustering(c_value, data)
    mistakes = 0
    for z in range(len(cluster)):
        if cluster[z] != np.round(dataset_[z][0]):
            mistakes += 1
    generalisation_error = (mistakes / len(cluster))
    return generalisation_error, c_value

if __name__ == '__main__':
    if not os.path.exists('plots/clustering'):
        os.makedirs('plots/clustering')

    marker = ["o", "x"]
    color = ['black', 'blue']
    dataset = get_data('twomoons.dat')
    # l = len(dataset)
    # n = len(dataset[0]-1)
    # X = dataset[:, 1:]
    # labels = dataset[:, :1]

    c_array = c_values(50, 0.1)

    # x = np.random.randint(1, 20, 5)
    # y = np.random.randint(1, 20, 5)
    # z = np.random.randint(1, 20, 5)
    # a = np.random.randint(1, 20, 5)
    # b = np.random.randint(1, 20, 5)
    # d = np.random.randint(1, 20, 5)
    # xyz = (x, (y, z, a, b, d))
    # print(xyz)
    # print(np.argsort(xyz[0])[1])
    # print(xyz[1][np.argsort(xyz[0])[1]])

    # generalisations = list(tuple())
    # for x in range((len(c_array))):
    #     cluster = clustering(c_array[x], X)
    #     mistakes = 0
    #     for i in range(len(cluster)):
    #         if cluster[i] != np.round(dataset[i][0]):
    #             mistakes += 1
    #     generalisation_error = (mistakes / len(cluster))
    #     generalisations.append((generalisation_error, np.round(np.round(np.arange(-50, 50.2, 0.1), 1))[x]))
    # print(generalisations)
    # print(min(generalisations))

    # print(calculate_cluster_error(-36.8, dataset))
    # print(calculate_cluster_error(-37, dataset))

    generalisations = list(tuple())
    p_bar = tqdm(smoothing=1)
    ppe = ProcessPoolExecutor(max_workers=4)
    for result in ppe.map(calculate_cluster_error, c_array, [dataset]*len(c_array)):
        generalisations.append(result)
        p_bar.update(1)

    generalisations.sort(key=lambda x: x[1])
    print(generalisations)
    print(min(generalisations))

    # plot_cluster(min(generalisations)[1], X)

    # l = 20
    # n = 10
    # c = 1
    # X = random_sample(n, l)
    # # print(X)
    # W = vectorised_weight(c, X)
    # D = vectorised_diagonalD(W)
    # L = D - W
    # # # print(W)
    # # # print(D)
    # # # print(L)
    # eigenspace = np.linalg.eig(L)
    # v_2 = eigenspace[1][1]
    # # # print(eigenspace[0])
    # # # print(eigenspace[1])
    # # print(vectorisedClustering(v_2))
    # print(clustering(v_2))
    # print(len(v_2))
    # df = pd.DataFrame({'x_values': X[:, 0], 'y_values': X[:, 1], 'labels': cluster})
