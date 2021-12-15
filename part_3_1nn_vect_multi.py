from concurrent.futures import ProcessPoolExecutor
import numpy as np
import part_3_perceptron as perceptron
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

np.random.seed(0)


def concurrent_optimised_sample_complexity(dimension):
    dimension_test = tuple()
    generalisation_error = 1
    sample_count = 1
    while generalisation_error >= 0.1:
        training_dataset, testing_dataset = perceptron.setup(dimension, sample_count)
        mistakes = 0
        for i in range(len(testing_dataset)):
            prediction = training_dataset[
                np.argmin(np.linalg.norm(testing_dataset[i][:-1] - training_dataset[:, :-1], axis=1))
            ][-1]
            if prediction != testing_dataset[i][-1]:
                mistakes += 1
        generalisation_error = mistakes / len(testing_dataset)
        dimension_test = (dimension, sample_count, generalisation_error)
        sample_count += 1
    # optimisation.append(dimension_test)
    return dimension_test


if __name__ == '__main__':
    if not os.path.exists('plots'):
        os.makedirs('plots')

    complexity = 20
    optimisation = list(tuple())
    p_bar = tqdm(smoothing=1)
    ppe = ProcessPoolExecutor(max_workers=2)
    for result in ppe.map(concurrent_optimised_sample_complexity, range(1, complexity + 1)):
        optimisation.append(result)
        p_bar.update(1)

    optimisation.sort(key=lambda x: x[0])
    print(optimisation)

    n_values = [n[0] for n in optimisation]
    m_values = [m[1] for m in optimisation]
    plt.plot(n_values, m_values)
    plt.title(f"1-NN generalisation error optimisation up to dimensionality of {str(complexity)}")
    plt.xlabel("dimension")
    plt.ylabel("sample size")
    plt.savefig(f'./plots/part3_a_1nn_{str(complexity)}_vect_mult.png')