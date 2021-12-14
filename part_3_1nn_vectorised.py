import numpy as np
import part_3_perceptron as perceptron
import matplotlib.pyplot as plt
# from concurrent import futures
from tqdm import tqdm
import os
# from line_profiler_pycharm import profile

np.random.seed(0)


# @profile
def predict(sample, dataset):
    # neighbour = nearest_neighbour(sample, dataset)
    neighbour = nearest_neighbour_vectorized(sample, dataset)
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


# @profile
def nearest_neighbour(sample, dataset):
    distances = []
    for point in dataset:
        temp_dist = calc_distance(sample[:-1], point[:-1])
        distances.append((point, temp_dist))
    distances.sort(key=lambda x: x[1])
    return distances[0][0]


# @profile
def nearest_neighbour_vectorized(sample, dataset):
    # distances_y = np.linalg.norm(sample[:-1] - dataset[:, :-1], axis=1)
    # min_index = np.argmin(distances_y)
    # to_return = dataset[min_index]  # debugging purposes
    # return to_return
    return dataset[np.argmin(np.linalg.norm(sample[:-1] - dataset[:, :-1], axis=1))]


def concurrent_optimised_sample_complexity(dimension, n):
    dimension_test = tuple()
    generalisation_error = 1
    sample_count = 1
    while generalisation_error >= 0.1:
        training_dataset, testing_dataset = perceptron.setup(dimension, sample_count)
        mistakes = 0
        for i in range(len(testing_dataset)):
            prediction = predict(testing_dataset[i], training_dataset)
            if prediction != testing_dataset[i][-1]:
                mistakes += 1
        generalisation_error = mistakes / len(testing_dataset)
        dimension_test = (dimension, sample_count, generalisation_error)
        sample_count += n
    optimisation.append(dimension_test)


if __name__ == '__main__':
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # complexity = 9
    n = 5
    complexity = 20
    optimisation = list(tuple())
    p_bar = tqdm(smoothing=1)
    # with futures.ThreadPoolExecutor(max_workers=1) as executor:
    #     futures_to_jobs = {executor.submit(concurrent_optimised_sample_complexity, job): job for job in
    #                        range(complexity + 1)}
    #     for _ in futures.as_completed(futures_to_jobs):
    #         p_bar.update(1)
    #         p_bar.refresh()
    for job in range(1, complexity + 1):
        concurrent_optimised_sample_complexity(job, n)
        p_bar.update(1)

    optimisation.sort(key=lambda x: x[0])
    print(optimisation)

    n_values = [n[0] for n in optimisation]
    m_values = [m[1] for m in optimisation]
    plt.plot(n_values, m_values)
    plt.title(f"1-NN generalisation error up to dimensionality of {str(complexity)}")
    plt.xlabel("dimension")
    plt.ylabel("sample size")
    plt.savefig(f'./plots/part3_a_1nn_{str(complexity)}_vectorised{str(n)}.png')