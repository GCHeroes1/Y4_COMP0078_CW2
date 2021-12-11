import numpy as np
import part_3_perceptron as perceptron

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
	for t in range (len(training_set[0])):
		for sample in training_set:
			y_hat_t = predict(sample, weights, n)
			y_t = sample[-1]
			if y_t != y_hat_t:
				for i in range (len(weights)):
					# print("made a mistake")
					weights[i] = weights[i] * 2 ** np.dot((y_t - y_hat_t), sample[i])
					# weights = weights + np.dot(y_t, sample)
				mistakes += 1
		# print(weights)
	return weights

def winnow(training_set, testing_set, n):
	predictions = np.zeros(len(testing_set))
	weights = train_weights(training_set, n)
	for i in range(len(testing_set)):
		prediction = predict(testing_set[i], weights, n)
		predictions[i] = prediction
	return predictions


if __name__ == '__main__':
	n = 10
	training_samples = 5
	testing_samples = 1000

	training_data = random_sample(n, training_samples)
	testing_data = random_sample(n, testing_samples)

	training_labels = perceptron.label(training_data)
	testing_labels = perceptron.label(testing_data)

	training_dataset = perceptron.create_dataset(training_data, training_labels)
	testing_dataset = perceptron.create_dataset(testing_data, testing_labels)

	# print(training_dataset)
	# weights = train_weights(training_dataset, n)

	predictions = winnow(training_dataset, testing_dataset, n)
	print(predictions)
	print(testing_labels)
	mistakes = 0
	for i in range(len(predictions)):
		if predictions[i] != testing_labels[i]:
			mistakes += 1
	print(mistakes / len(predictions))
	# print(weights)

