import numpy as np
from tqdm import tqdm
from utils import get_data, split_indices_with_shuffle, plot_accuracy
from kernels import PolynomialKernel
from perceptrons import Perceptron

X_data, Y_data = get_data("../zipcombo.dat")

# K-Class Classifier Parameters:
N_CLASSES = 10
N_CLASSIFIERS = int(N_CLASSES * (N_CLASSES - 1) / 2)
N_EPOCHS = 20
RUNS = 20
PLOT_ACCURACY = True
BASE_DIR = '../plots/part1/q6_1'

classifier_values = np.zeros((N_CLASSIFIERS, 2))
idx = 0
for c1 in np.arange(10):
	for c2 in np.arange(c1 + 1, 10):
		classifier_values[idx] = [c1, c2]
		idx += 1

train_errors_mean = []
train_errors_std = []
test_errors_mean = []
test_errors_std = []

# iterate through all d values from 1 to 7
for d in tqdm(range(1, 8)):
	print(f"d = {d}")

	train_error_per_run = []
	test_error_per_run = []

	# kernelize data
	kernel_class = PolynomialKernel(d)
	kernelized_matrix = kernel_class.apply_kernel(X_data)

	# 20 runs
	for run in range(1, RUNS + 1):
		print(f"Run: {run}")

		# split indices with shuffling
		train_indices, test_indices = split_indices_with_shuffle(Y_data)

		# train test split
		X_train = X_data[train_indices]
		Y_train = Y_data[train_indices]
		X_test = X_data[test_indices]
		Y_test = Y_data[test_indices]

		n_train_samples = len(train_indices)
		n_test_samples = len(test_indices)

		# initiate perceptron model class
		alphas = np.zeros((N_CLASSIFIERS, n_train_samples))
		ovo_model = Perceptron(alphas, kernelized_matrix, N_CLASSES, method='ovo', classifier_values=classifier_values)

		# save results
		train_corrects = []
		train_accuracies = []
		train_mistakes = []
		test_corrects = []
		test_accuracies = []
		test_mistakes = []

		for epoch in range(1, N_EPOCHS + 1):
			alphas = ovo_model.fit(train_indices, X_train, Y_train)

			# GET TRAIN ACCURACY
			kernel_vals = kernelized_matrix[train_indices][:, train_indices]
			pred = ovo_model.predict_ovo(alphas, kernel_vals)
			train_correct = len(np.where((Y_train - pred.astype(int)) == 0)[0])

			# Record results:
			train_corrects.append(train_correct)

			train_accuracy = train_correct / n_train_samples
			train_accuracies.append(train_accuracy)

			train_mistake = n_train_samples - train_correct
			train_mistakes.append(train_mistake)

			# GET TEST ACCURACY
			kernel_vals = kernelized_matrix[train_indices][:, test_indices]
			pred = ovo_model.predict_ovo(alphas, kernel_vals)
			test_correct = len(np.where((Y_test - pred.astype(int)) == 0)[0])

			# Record results:
			test_corrects.append(test_correct)

			test_accuracy = test_correct / n_test_samples
			test_accuracies.append(test_accuracy)

			test_mistake = n_test_samples - test_correct
			test_mistakes.append(test_mistake)

		# save error rate
		train_error_per_run.append(train_mistake / n_train_samples)
		test_error_per_run.append(test_mistake / n_test_samples)

		if PLOT_ACCURACY:
			# plot accuracy graph
			plot_accuracy(d, run, N_EPOCHS + 1, train_accuracies, test_accuracies, save_dir=BASE_DIR)

	# mean of errors and std:
	train_errors_mean.append(np.mean(train_error_per_run))
	train_errors_std.append(np.std(train_error_per_run))
	test_errors_mean.append(np.mean(test_error_per_run))
	test_errors_std.append(np.std(test_error_per_run))

with open('../plots/part1/q6_1/log.txt', 'w') as f:
	f.write(f"Mean train error: {train_errors_mean}\n")
	f.write(f"STD training error: {train_errors_std}\n")
	f.write(f"Mean testing error: {test_errors_mean}\n")
	f.write(f"STD testing error: {test_errors_std}\n")
	f.close()
