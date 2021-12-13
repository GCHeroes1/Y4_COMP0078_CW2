import numpy as np
import part_3_perceptron as perceptron
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue
import multiprocessing as mp
from tqdm import tqdm
import os

np.random.seed(0)

def predict(sample, dataset):
    neighbour = nearest_neighbour(sample, dataset)
    label = neighbour[-1]
    return label

def calc_distance(x1, x2):
    dist = 0
    for i in range(len(x1)):
        dist += np.square(x1[i] - x2[i])
    return np.sqrt(dist)

def calc_distance_array(x1, x2):
    dist = np.sum(np.square(x1-x2))
    return np.sqrt(dist)

def calc_distance_array_2(x1, x2):
    return np.linalg.norm(x1 - x2)

def nearest_neighbour(sample, dataset):
    distances = []
    for point in dataset:
        temp_dist = calc_distance_array_2(sample[:-1], point[:-1])
        distances.append((point, temp_dist))
    distances.sort(key=lambda x:x[1])
    return distances[0][0]

def concurrent_optimised_sample_complexity(queue, dimension):
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
        generalisation_error = mistakes/len(testing_dataset)
        dimension_test = (dimension, sample_count, generalisation_error)
        sample_count += 5
    queue.put(dimension_test)


if __name__ == '__main__':
    if not os.path.exists('plots'):
        os.makedirs('plots')

    complexity = 20
    optimisation = list(tuple())
    p_bar = tqdm(complexity)

    # pool = mp.Pool(mp.cpu_count())
    # pool.map(concurrent_optimised_sample_complexity, range(1, complexity+1))

    queue = Queue()
    processes = [Process(target=concurrent_optimised_sample_complexity, args=(queue, x)) for x in range(1, complexity+1)]

    for p in processes:
        p.start()
    for p in processes:
        p.join()
        p_bar.update(1)

    optimisation = [queue.get() for p in processes]

    # with futures.ThreadPoolExecutor(max_workers=12) as executor:
    #     futures_to_jobs = {executor.submit(concurrent_optimised_sample_complexity, job): job for job in
    #                        range(complexity + 1)}
    #     for _ in futures.as_completed(futures_to_jobs):
    #         p_bar.update(1)
    #         p_bar.refresh()

    optimisation.sort(key=lambda x: x[0])
    print(optimisation)

    n_values = [n[0] for n in optimisation]
    m_values = [m[1] for m in optimisation]
    plt.plot(n_values, m_values)
    plt.title(f"1-NN generalisation error up to dimensionality of {str(complexity)}")
    plt.xlabel("dimension")
    plt.ylabel("sample size")
    plt.savefig(f'./plots/part3_a_1nn_{str(complexity)}_multiprocess.png')

    # n = 4
    # training_samples = 10
    # testing_samples = 1000
    #
    # training_data = perceptron.random_sample(n, training_samples)
    # testing_data = perceptron.random_sample(n, testing_samples)
    #
    # training_labels = perceptron.label(training_data)
    # testing_labels = perceptron.label(testing_data)
    #
    # training_dataset = perceptron.create_dataset(training_data, training_labels)
    # testing_dataset = perceptron.create_dataset(testing_data, testing_labels)
    # print(testing_dataset[1][:-1])
    # # distance = []
    # # for i in range(len(training_dataset[:-1])-1):
    # #     for x in range(len(training_dataset[:-1])-1):
    # #         distance.append(calc_distance(training_dataset[i], training_dataset[x]))
    # # print(max(distance))
    #
    # closest_neighbour = nearest_neighbour(testing_dataset[1], training_dataset)
    # print(closest_neighbour)
    # print(calc_distance(closest_neighbour[:-1], testing_dataset[1][:-1]))
    #
    # mistakes = 0
    # for i in range(len(testing_dataset)):
    #     prediction = predict(testing_dataset[i], training_dataset)
    #     if prediction != testing_dataset[i][-1]:
    #         mistakes += 1
    # print(mistakes/len(testing_dataset))
