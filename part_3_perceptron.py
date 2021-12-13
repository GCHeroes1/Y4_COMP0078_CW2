import numpy as np
import matplotlib.pyplot as plt
import os
from concurrent import futures
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

def sample_complexity(dimensions, training_samples):
	# okay one thing ill do is keep testing samples constant, because five fold validation would mean that the testing
	# scales with the number of samples
	optimal_generalisation_error = []
	for i in range(1, dimensions):
		dimension_test = []
		for x in range(1, training_samples):
			training_dataset, testing_dataset = setup(i, x)
			predictions = perceptron(training_dataset, testing_dataset)
			mistakes = 0
			for z in range(len(predictions)):
				if predictions[z] != testing_dataset[z][-1]:
					mistakes += 1
			generalisation_error = mistakes/len(predictions)
			dimension_test.append((i, x, generalisation_error))
		dimension_test.sort(key=lambda test: test[2])
		optimal_generalisation_error.append(dimension_test[0])
	return optimal_generalisation_error


def optimised_sample_complexity(dimensions):
	optimal_generalisation_error = []
	for i in range(1, dimensions):
		dimension_test = (0, 0, 0)
		generalisation_error = 1
		sample_count = 1
		while generalisation_error >= 0.1:
			training_dataset, testing_dataset = setup(i, sample_count)
			predictions = perceptron(training_dataset, testing_dataset)
			mistakes = 0
			for z in range(len(predictions)):
				if predictions[z] != testing_dataset[z][-1]:
					mistakes += 1
			generalisation_error = mistakes/len(predictions)
			dimension_test = (i, sample_count, generalisation_error)
			sample_count += 1
		optimal_generalisation_error.append(dimension_test)
	return optimal_generalisation_error

def concurrent_optimised_sample_complexity(dimension):
	dimension_test = tuple()
	generalisation_error = 1
	sample_count = 1
	while generalisation_error >= 0.1:
		training_dataset, testing_dataset = setup(dimension, sample_count)
		predictions = perceptron(training_dataset, testing_dataset)
		mistakes = 0
		for z in range(len(predictions)):
			if predictions[z] != testing_dataset[z][-1]:
				mistakes += 1
		generalisation_error = mistakes/len(predictions)
		dimension_test = (dimension, sample_count, generalisation_error)
		sample_count += 1
	optimisation.append(dimension_test)

if __name__ == '__main__':
	if not os.path.exists('plots'):
		os.makedirs('plots')

	complexity = 200
	optimisation = list(tuple())
	p_bar = tqdm(complexity)
	with futures.ThreadPoolExecutor(max_workers=12) as executor:
		futures_to_jobs = {executor.submit(concurrent_optimised_sample_complexity, job): job for job in range(complexity+1)}
		for _ in futures.as_completed(futures_to_jobs):
			p_bar.update(1)

	optimisation.sort(key=lambda x:x[0])
	print(optimisation)

	n_values = [n[0] for n in optimisation]
	m_values = [m[1] for m in optimisation]
	plt.plot(n_values, m_values)
	plt.title(f"Perceptron generalisation error up to dimensionality of {str(complexity)}")
	plt.xlabel("dimension")
	plt.ylabel("sample size")
	plt.savefig(f'./plots/part3_a_perceptron_{str(complexity)}.png')

	# n = 10
	# training_samples = 5
	# testing_samples = 1000
	#
	# training_data = random_sample(n, training_samples)
	# testing_data = random_sample(n, testing_samples)
	#
	# training_labels = label(training_data)
	# testing_labels = label(testing_data)
	#
	# training_dataset = create_dataset(training_data, training_labels)
	# testing_dataset = create_dataset(testing_data, testing_labels)

	# weights = train_weights(training_dataset)
	# predictions = perceptron(training_dataset, testing_dataset)
	# print(predictions)
	# print(testing_labels)

	# mistakes = 0
	# for i in range(len(predictions)):
	# 	if predictions[i] != testing_labels[i]:
	# 		mistakes += 1
	# print(mistakes/len(predictions))
	# print(weights)

