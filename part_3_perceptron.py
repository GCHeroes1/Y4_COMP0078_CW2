import numpy as np

# perceptron
# winnow
# least squares
# 1-nearest neighbours

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
				# print("made a mistake")
				weights = weights + np.dot(y_t, sample)
				mistakes += 1
		# print(weights)
	return weights

def perceptron(training_set, testing_set):
	# assume linearly separable data
	# split classes, nearest 2 points is atleast gamma
	# v is vector corresponding to linear classifier
	# assume all data lives in radius R around origin, y_t matches prediction of linear predictor v and the gap is larger than gamma
	# get data sequence, initialise weight to w, use m to count number of mis-classifications.
	# If we make a mistake, update internal state of algorithm, update weight vector, add new example multipled by y_t to weight vector, record that we incurred mistake
	predictions = np.zeros(len(testing_set))
	weights = train_weights(training_set)
	for i in range(len(testing_set)):
		prediction = predict(testing_set[i], weights)
		predictions[i] = prediction
	return predictions

if __name__ == '__main__':
	n = 10
	training_samples = 5
	testing_samples = 1000

	training_data = random_sample(n, training_samples)
	testing_data = random_sample(n, testing_samples)

	training_labels = label(training_data)
	testing_labels = label(testing_data)

	training_dataset = create_dataset(training_data, training_labels)
	testing_dataset = create_dataset(testing_data, testing_labels)

	# weights = train_weights(training_dataset)
	predictions = perceptron(training_dataset, testing_dataset)
	print(predictions)
	print(testing_labels)
	mistakes = 0
	for i in range(len(predictions)):
		if predictions[i] != testing_labels[i]:
			mistakes += 1
	print(mistakes/len(predictions))
	# print(weights)

