import numpy as np
import matplotlib.pyplot as plt
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

np.random.seed(0)

def random_sample(dimension, n_times):
    data = np.random.choice((-1, 1), n_times*dimension)
    return np.reshape(data, (n_times, dimension))

def label(data):
    y = np.zeros(len(data))
    for i in range(len(data)):
        y[i] = data[i][0]
    return y

def create_dataset(data, label):
    dataset_ = np.zeros((len(data), len(data[0]) + 1))
    for i in range(len(data)):
        entry = np.append(data[i], label[i])
        dataset_[i] = entry
    return dataset_

def predict(sample, weights):
    y_hat_t = np.dot(sample, weights)
    if y_hat_t >= 0:
        return 1
    else:
        return -1

def train_weights(training_set):
    weights = np.zeros(len(training_set[0]))
    mistakes = 0
    for t in range (len(training_set[0])):
        for sample in training_set:
            y_hat_t = predict(sample, weights)
            y_t = sample[-1]
            mistake = y_hat_t * y_t
            if mistake <= 0:
                weights = weights + np.dot(y_t, sample)
                mistakes += 1
    return weights

def perceptron(training_set, testing_set):
    predictions = np.zeros(len(testing_set))
    weights = train_weights(training_set)
    for i in range(len(testing_set)):
        prediction = predict(testing_set[i], weights)
        predictions[i] = prediction
    return predictions

def setup(n, m):
    testing_samples_ = 1000

    training_data_ = random_sample(n, m)
    testing_data_ = random_sample(n, testing_samples_)

    training_labels_ = label(training_data_)
    testing_labels_ = label(testing_data_)

    training_dataset_ = create_dataset(training_data_, training_labels_)
    testing_dataset_ = create_dataset(testing_data_, testing_labels_)

    return training_dataset_, testing_dataset_

def dataset_setup(n, m):
    data_ = random_sample(n, m)
    labels_ = label(data_)
    dataset_ = create_dataset(data_, labels_)
    return dataset_

def average_sample_complexity(dimension):
    sample_sizes = list()
    for z in range(10):
        sample_count = 1
        generalisation_error = 1
        testing_dataset = dataset_setup(dimension, 1000)
        while generalisation_error >= 0.1:
            training_dataset = dataset_setup(dimension, sample_count)
            predictions = perceptron(training_dataset, testing_dataset)
            mistakes = 0
            for z in range(len(predictions)):
                if predictions[z] != testing_dataset[z][-1]:
                    mistakes += 1
            generalisation_error = (mistakes / len(testing_dataset))
            sample_count += 1
        sample_sizes.append(sample_count)
    sample_average = np.average(sample_sizes)
    sample_std = np.std(sample_sizes)
    return dimension, sample_average, sample_std


if __name__ == '__main__':
    if not os.path.exists('plots/perceptron'):
        os.makedirs('plots/perceptron')

    n = 1
    complexity = 100
    optimisation = list(tuple())
    p_bar = tqdm(smoothing=1)
    ppe = ProcessPoolExecutor(max_workers=2)
    for result in ppe.map(average_sample_complexity, range(1, complexity + 1)):
        optimisation.append(result)
        p_bar.update(1)

    optimisation.sort(key=lambda x:x[0])
    print(optimisation)

    n_values = [n[0] for n in optimisation]
    m_values = [m[1] for m in optimisation]
    error = [err[2] for err in optimisation]
    plt.plot(n_values, m_values, color='black')
    eb = plt.errorbar(n_values, m_values, error, ecolor='orange')
    plt.title(f"Perceptron generalisation error up to dimensionality of {str(complexity)}")
    plt.xlabel("dimension")
    plt.ylabel("sample size")
    plt.savefig(f'./plots/perceptron/part3_a_perceptron_{str(complexity)}_{str(n)}_error.png')

