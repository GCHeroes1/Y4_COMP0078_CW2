import numpy as np
import part_3_perceptron as perceptron
import matplotlib.pyplot as plt
from concurrent import futures
from tqdm import tqdm
import os

np.random.seed(0)

def random_sample(dimension, n_times):
	data = np.random.choice((0, 1), n_times*dimension)
	return np.reshape(data, (n_times, dimension))

def predict(sample, weights, n):
	y_hat_t = np.dot(sample, weights)
	if y_hat_t < n:
		return 0
	else:
		return 1

def train_weights(training_set, n):
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
	predictions = np.zeros(len(testing_set))
	weights = train_weights(training_set, n)
	for i in range(len(testing_set)):
		prediction = predict(testing_set[i], weights, n)
		predictions[i] = prediction
	return predictions

def setup(n, m):
	testing_samples_ = 1000

	training_data_ = random_sample(n, m)
	testing_data_ = random_sample(n, testing_samples_)

	training_labels_ = perceptron.label(training_data_)
	testing_labels_ = perceptron.label(testing_data_)

	training_dataset_ = perceptron.create_dataset(training_data_, training_labels_)
	testing_dataset_ = perceptron.create_dataset(testing_data_, testing_labels_)

	return training_dataset_, testing_dataset_

def concurrent_optimised_sample_complexity(dimension):
	dimension_test = tuple()
	generalisation_error = 1
	sample_count = 1
	while generalisation_error >= 0.1:
		training_dataset, testing_dataset = setup(dimension, sample_count)
		predictions = winnow(training_dataset, testing_dataset, dimension)
		mistakes = 0
		for z in range(len(predictions)):
			if predictions[z] != testing_dataset[z][-1]:
				mistakes += 1
		generalisation_error = mistakes/len(predictions)
		dimension_test = (dimension, sample_count, generalisation_error)
		sample_count += 1
	# return
	optimisation.append(dimension_test)


if __name__ == '__main__':

	if not os.path.exists('plots'):
		os.makedirs('plots')

	complexity = 500
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
	plt.title(f"Winnow generalisation error optimisation up to dimensionality of {str(complexity)}")
	plt.xlabel("dimension")
	plt.ylabel("sample size")
	plt.savefig(f'./plots/part3_a_winnow_{str(complexity)}.png')

	# n = 3
	# training_samples = 5
	# testing_samples = 1000
	#
	# training_data = random_sample(n, training_samples)
	# testing_data = random_sample(n, testing_samples)
	#
	# training_labels = perceptron.label(training_data)
	# testing_labels = perceptron.label(testing_data)
	#
	# training_dataset = perceptron.create_dataset(training_data, training_labels)
	# testing_dataset = perceptron.create_dataset(testing_data, testing_labels)
	#
	# # print(training_dataset)
	# # weights = train_weights(training_dataset, n)
	#
	# predictions = winnow(training_dataset, testing_dataset, n)
	# print(predictions)
	# print(testing_labels)
	# mistakes = 0
	# for i in range(len(predictions)):
	# 	if predictions[i] != testing_labels[i]:
	# 		mistakes += 1
	# print(mistakes / len(predictions))
	# # print(weights)

