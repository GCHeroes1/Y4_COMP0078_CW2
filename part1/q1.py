"""
Basic Results: Perform 20 runs for d = 1, . . . , 7 each run should randomly split zipcombo into 80% train and 20%
test. Report the mean test and train error rates as well as well as standard deviations. Thus your data table, here,
will be 2 × 7 with each “cell” containing a mean±std.
"""
import numpy as np
from tqdm import tqdm
from utils import get_data, split_indices_with_shuffle, plot_accuracy
from kernels import PolynomialKernel
from perceptrons import Perceptron

N_CLASSES = 10
N_EPOCHS = 1
RUNS = 20
PLOT_ACCURACY = False

# get data as X and Y
X_data, Y_data = get_data("../zipcombo.dat")

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
		alphas = np.zeros((N_CLASSES, n_train_samples))
		model = Perceptron(alphas, kernelized_matrix, N_CLASSES)

		# save results
		train_corrects = []
		train_accuracies = []
		train_mistakes = []
		test_corrects = []
		test_accuracies = []
		test_mistakes = []

		for epoch in range(1, N_EPOCHS + 1):
			alphas = model.fit(train_indices, X_train, Y_train)

			# GET TRAIN ACCURACY
			kernel_vals = kernelized_matrix[train_indices][:, train_indices]
			pred = np.array(list(map(np.argmax, zip(*(alphas.dot(kernel_vals))))))
			train_correct = len(np.where((Y_train - pred) == 0)[0])

			# Record results:
			train_corrects.append(train_correct)

			train_accuracy = train_correct / n_train_samples
			train_accuracies.append(train_accuracy)

			train_mistake = n_train_samples - train_correct
			train_mistakes.append(train_mistake)

			# GET TEST ACCURACY
			kernel_vals = kernelized_matrix[train_indices][:, test_indices]
			pred = np.array(list(map(np.argmax, zip(*(alphas.dot(kernel_vals))))))
			test_correct = len(np.where((Y_test - pred) == 0)[0])

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
			plot_accuracy(d, run, N_EPOCHS + 1, train_accuracies, test_accuracies)

	# mean of errors and std:
	train_errors_mean.append(np.mean(train_error_per_run))
	train_errors_std.append(np.std(train_error_per_run))
	test_errors_mean.append(np.mean(test_error_per_run))
	test_errors_std.append(np.std(test_error_per_run))

with open('../plots/part1/q1/log.txt', 'w') as f:
	f.write(f"Mean train error: {train_errors_mean}\n")
	f.write(f"STD training error: {train_errors_std}\n")
	f.write(f"Mean testing error: {test_errors_mean}\n")
	f.write(f"STD testing error: {test_errors_std}\n")
	f.close()
