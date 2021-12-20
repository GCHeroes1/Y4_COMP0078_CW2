import numpy as np
import part_3_perceptron as perceptron
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import os

np.random.seed(0)


def random_sample(dimension, n_times):
	data = np.random.choice((0, 1), n_times * dimension)
	return np.reshape(data, (n_times, dimension))


def predict(sample, weights, n):
	y_hat_t = np.dot(sample, weights)
	if y_hat_t < n:
		return 0
	else:
		return 1


def train_weights(training_set, n):
	# training weights by generating predictions and updating weights if a mistake was made
	weights = np.ones(len(training_set[0]))
	mistakes = 0
	for t in range(len(training_set[0])):
		for sample in training_set:
			y_hat_t = predict(sample, weights, n)
			y_t = sample[-1]
			if y_t != y_hat_t:
				for i in range(len(weights)):
					weights[i] = weights[i] * 2 ** np.dot((y_t - y_hat_t), sample[i])
				mistakes += 1
	return weights


def winnow(training_set, testing_set, n):
	# main winnow algorithm, firstly train the weights and then making predictions on the testing set to be used later
	predictions = np.zeros(len(testing_set))
	weights = train_weights(training_set, n)
	for i in range(len(testing_set)):
		prediction = predict(testing_set[i], weights, n)
		predictions[i] = prediction
	return predictions


def dataset_setup(n, m):
	data_ = random_sample(n, m)
	labels_ = perceptron.label(data_)
	dataset_ = perceptron.create_dataset(data_, labels_)
	return dataset_


def average_sample_complexity(dimension):
	"""
	create the testing_dataset initially to test the trained weights on, generate predictions for each training and
	testing set, check the generalisation error for the training and repeat until 10% error is achieved (increasing
	training set size in order to improve training), once achieved, repeat the same test again with a new testing
	dataset to eventually calculate the mean and standard deviation of sample size required to achieve a 10%
	generalisation error for the current dimension.
	:param dimension: dimension defines the dimensionality of the data points
	:return: return the dimension (index for plotting), and sample average (points to plot), and sample_std (error bars)
	"""
	sample_sizes = list()
	for z in range(10):
		sample_count = 1
		generalisation_error = 1
		testing_dataset = dataset_setup(dimension, 1000)
		while generalisation_error >= 0.1:
			training_dataset = dataset_setup(dimension, sample_count)
			predictions = winnow(training_dataset, testing_dataset, dimension)
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

	if not os.path.exists('plots/winnow'):
		os.makedirs('plots/winnow')

	n = 1
	complexity = 100
	optimisation = list(tuple())
	p_bar = tqdm(smoothing=1)
	ppe = ProcessPoolExecutor(max_workers=4)
	for result in ppe.map(average_sample_complexity, range(1, complexity + 1), [n] * complexity):
		optimisation.append(result)
		p_bar.update(1)

	optimisation.sort(key=lambda x: x[0])
	print(optimisation)

	n_values = [n[0] for n in optimisation]
	m_values = [m[1] for m in optimisation]
	error = [err[2] for err in optimisation]
	plt.plot(n_values, m_values, color='black')
	eb = plt.errorbar(n_values, m_values, error, ecolor='orange')
	plt.title(f"Winnow generalisation error up to dimensionality of {str(complexity)}")
	plt.xlabel("dimension")
	plt.ylabel("sample size")
	plt.savefig(f'./plots/winnow/part3_a_winnow_{str(complexity)}_{str(n)}_error_1000.png')
