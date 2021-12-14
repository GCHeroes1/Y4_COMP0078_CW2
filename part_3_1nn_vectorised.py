import numpy as np
import part_3_perceptron as perceptron
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

np.random.seed(0)


def predict(sample, dataset):
    neighbour = nearest_neighbour(sample, dataset)
    # neighbour = dataset[np.argmin(np.linalg.norm(sample[:-1] - dataset[:, :-1], axis=1))]
    label = neighbour[-1]
    return label


def calc_distance(x1, x2):
    dist = 0
    for i in range(len(x1)):
        dist += np.square(x1[i] - x2[i])
    return np.sqrt(dist)


def calc_distance_array(x1, x2):
    dist = np.sum(np.square(x1 - x2))
    return np.sqrt(dist)


def nearest_neighbour(sample, dataset):
    distances = []
    for point in dataset:
        temp_dist = calc_distance(sample[:-1], point[:-1])
        distances.append((point, temp_dist))
    distances.sort(key=lambda x: x[1])
    return distances[0][0]


def nearest_neighbour_vectorized(sample, dataset):
    # distances_y = np.linalg.norm(sample[:-1] - dataset[:, :-1], axis=1)
    # min_index = np.argmin(distances_y)
    # to_return = dataset[min_index]  # debugging purposes
    # return to_return
    return dataset[np.argmin(np.linalg.norm(sample[:-1] - dataset[:, :-1], axis=1))]


def concurrent_optimised_sample_complexity(dimension, count):
    dimension_test = tuple()
    generalisation_error = 1
    sample_count = 1
    while generalisation_error >= 0.1:
        training_dataset, testing_dataset = perceptron.setup(dimension, sample_count)
        mistakes = 0
        for i in range(len(testing_dataset)):
            # prediction = training_dataset[np.argmin(np.linalg.norm(testing_dataset[i][:-1] - training_dataset[:, :-1],
            #                                                        axis=1))][-1]
            # uncomment for performance
            prediction = predict(testing_dataset[i], training_dataset)
            if prediction != testing_dataset[i][-1]:
                mistakes += 1
        generalisation_error = mistakes / len(testing_dataset)
        dimension_test = (dimension, sample_count, generalisation_error)
        sample_count += count
    optimisation.append(dimension_test)


if __name__ == '__main__':
    if not os.path.exists('plots'):
        os.makedirs('plots')

    complexity = 20
    sample_count = 2
    optimisation = list(tuple())
    p_bar = tqdm(smoothing=1)
    for job in range(1, complexity + 1):
        concurrent_optimised_sample_complexity(job, sample_count)
        p_bar.update(1)

    optimisation.sort(key=lambda x: x[0])
    print(optimisation)

    n_values = [n[0] for n in optimisation]
    m_values = [m[1] for m in optimisation]
    plt.plot(n_values, m_values)
    plt.title(f"1-NN generalisation error up to dimensionality of {str(complexity)}")
    plt.xlabel("dimension")
    plt.ylabel("sample size")
    plt.savefig(f'./plots/part3_a_1nn_{str(complexity)}_vectorised_{str(sample_count)}.png')