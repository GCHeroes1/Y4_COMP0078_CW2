import numpy as np
import part_3_perceptron as perceptron
import matplotlib.pyplot as plt
from concurrent import futures
from tqdm import tqdm
import os

np.random.seed(0)

# least squares, find function f(x) = np.dot(w.T, x) which best fits data
# prediction is y_i = np.dot(w.T, x_i)

def concurrent_optimised_sample_complexity(dimension):
    dimension_test = tuple()
    generalisation_error = 1
    sample_count = 1
    while generalisation_error >= 0.1:
        training_dataset, testing_dataset = perceptron.setup(dimension, sample_count)
        training_data = [data[:-1] for data in training_dataset]
        training_labels = [label[-1] for label in training_dataset]
        weights = np.dot(np.linalg.pinv(training_data), training_labels)
        mistakes = 0
        for i in range(len(testing_dataset)):
            prediction = np.dot(testing_dataset[i][:-1], weights)
            rounded_pred = np.round(prediction)
            if rounded_pred != testing_dataset[i][-1]:
                mistakes += 1
        generalisation_error = mistakes / len(testing_dataset)
        dimension_test = (dimension, sample_count, generalisation_error)
        sample_count += 1
    optimisation.append(dimension_test)


if __name__ == '__main__':
    if not os.path.exists('plots'):
        os.makedirs('plots')

    complexity = 20
    optimisation = list(tuple())
    p_bar = tqdm(complexity)
    with futures.ThreadPoolExecutor(max_workers=12) as executor:
        futures_to_jobs = {executor.submit(concurrent_optimised_sample_complexity, job): job for job in
                           range(complexity + 1)}
        for _ in futures.as_completed(futures_to_jobs):
            p_bar.update(1)

    optimisation.sort(key=lambda x: x[0])
    print(optimisation)

    n_values = [n[0] for n in optimisation]
    m_values = [m[1] for m in optimisation]
    plt.plot(n_values, m_values)
    plt.title(f"Least squares generalisation error up to dimensionality of {str(complexity)}")
    plt.xlabel("dimension")
    plt.ylabel("sample size")
    plt.savefig(f'./plots/part3_a_LSE_{str(complexity)}.png')

    # n = 100
    # training_samples = 400
    # testing_samples = 100
    #
    # training_data = perceptron.random_sample(n, training_samples)
    # testing_data = perceptron.random_sample(n, testing_samples)
    #
    # training_labels = perceptron.label(training_data)
    # testing_labels = perceptron.label(testing_data)
    #
    # training_dataset = perceptron.create_dataset(training_data, training_labels)
    # testing_dataset = perceptron.create_dataset(testing_data, testing_labels)
    #
    # weights = np.dot(np.linalg.pinv(training_data), training_labels)
    #
    # mistakes = 0
    # for i in range(len(testing_dataset)):
    #     prediction = np.dot(testing_dataset[i][:-1], weights)
    #     rounded_pred = np.round(prediction)
    #     if rounded_pred != testing_dataset[i][-1]:
    #         mistakes += 1
    # print(mistakes/len(testing_dataset))
