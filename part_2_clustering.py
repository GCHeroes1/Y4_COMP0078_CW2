import numpy as np
import matplotlib.pyplot as plt
import os
import time


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
    np.fill_diagonal(D, np.sum(weights, axis=1))
    return D

def diagonalD(weights):
    D = np.zeros((len(weights), len(weights)))
    for i in range(len(weights)):
        D[i][i] = np.sum(weights, axis=1)[i]
    return D


def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0


def clustering(c, x):

    weight_matrix = vectorised_weight(c, x)
    diagonal_matrix = vectorised_diagonalD(weight_matrix)
    laplacian = diagonal_matrix - weight_matrix
    eigenspace = np.linalg.eig(laplacian)
    v_2 = eigenspace[1][1]
    print(len(v_2))

    cluster = np.zeros(len(v_2))
    for i in range(len(v_2)):
        cluster[i] = sign(v_2[i])
    return cluster


def vectorisedClustering(c, x):
# not vectorised yet
    weight_matrix = vectorised_weight(c, x)
    diagonal_matrix = vectorised_diagonalD(weight_matrix)
    laplacian = diagonal_matrix - weight_matrix
    eigenspace = np.linalg.eig(laplacian)
    v_2 = eigenspace[1][1]
    # print(len(v_2))

    # cluster = np.fromfunction(sign, v)
    return cluster


if __name__ == '__main__':
    marker = ["o", "x"]
    color = ['r', 'b']
    dataset = get_data('twomoons.dat')
    l = len(dataset)
    n = len(dataset[0]-1)
    X = dataset[:, 1:]
    labels = dataset[:, :1]
    c = 2**5
    cluster = clustering(c, X)

    for i in range(len(cluster)):
        temp_marker, temp_color = marker[1], color[1]
        if cluster[i] == 1:
            temp_marker, temp_color = marker[0], color[0]
        plt.scatter(X[i, 0], X[i, 1], marker=temp_marker, color=temp_color)
    plt.show()

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
