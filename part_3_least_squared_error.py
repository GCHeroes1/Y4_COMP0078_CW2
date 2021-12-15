import numpy as np
import part_3_perceptron as perceptron
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import os

np.random.seed(0)

# least squares, find function f(x) = np.dot(w.T, x) which best fits data
# prediction is y_i = np.dot(w.T, x_i)


def average_sample_complexity(dimension):
    sample_sizes = list()
    for z in range(10):
        sample_count = 1
        generalisation_error = 1
        testing_dataset = perceptron.dataset_setup(dimension, 1000)
        while generalisation_error >= 0.1:
            training_dataset = perceptron.dataset_setup(dimension, sample_count)
            training_data = [data[:-1] for data in training_dataset]
            training_labels = [label[-1] for label in training_dataset]
            weights = np.dot(np.linalg.pinv(training_data), training_labels)
            mistakes = 0
            for i in range(len(testing_dataset)):
                prediction = np.dot(testing_dataset[i][:-1], weights)
                rounded_pred = np.round(prediction)
                if rounded_pred != testing_dataset[i][-1]:
                    mistakes += 1
            generalisation_error = (mistakes / len(testing_dataset))
            sample_count += 6
        sample_sizes.append(sample_count)
    sample_average = np.average(sample_sizes)
    sample_std = np.std(sample_sizes)
    return dimension, sample_average, sample_std


if __name__ == '__main__':
    if not os.path.exists('plots/LSE'):
        os.makedirs('plots/LSE')

    n = 6
    complexity = 100
    optimisation = list(tuple())
    p_bar = tqdm(smoothing=1)
    ppe = ProcessPoolExecutor(max_workers=2)
    for result in ppe.map(average_sample_complexity, range(1, complexity + 1), [n]*complexity):
        optimisation.append(result)
        p_bar.update(1)

    optimisation.sort(key=lambda x: x[0])
    print(optimisation)

    n_values = [n[0] for n in optimisation]
    m_values = [m[1] for m in optimisation]
    error = [err[2] for err in optimisation]
    plt.plot(n_values, m_values, color='black')
    eb = plt.errorbar(n_values, m_values, error, ecolor='orange')
    plt.title(f"Least squares generalisation error up to dimensionality of {str(complexity)}")
    plt.xlabel("dimension")
    plt.ylabel("sample size")
    plt.savefig(f'./plots/LSE/part3_a_LSE_{str(complexity)}_{str(n)}_error.png')
