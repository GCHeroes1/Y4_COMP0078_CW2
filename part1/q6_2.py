import numpy as np
from tqdm import tqdm
from utils import get_data, plot_accuracy, five_fold_split, plot_digit, split_indices_with_shuffle
from kernels import PolynomialKernel
from perceptrons import Perceptron

X_data, Y_data = get_data("../zipcombo.dat")

N_CLASSES = 10
N_EPOCHS = 2
RUNS = 20
N_CLASSIFIERS = int(N_CLASSES * (N_CLASSES - 1) / 2)

classifier_values = np.zeros((N_CLASSIFIERS, 2))
idx = 0
for c1 in np.arange(10):
	for c2 in np.arange(c1 + 1, 10):
		classifier_values[idx] = [c1, c2]
		idx += 1

# q2
testing_errors = []
d_stars = []

# Loop for 20 runs
for run in tqdm(range(1, RUNS + 1)):
	print(f"Run: {run}")

	# split indices with shuffling
	train_indices, test_indices = split_indices_with_shuffle(Y_data)

	# Loop for d from 1 to 7
	avg_validation_correct = [0]
	for d in range(1, 8):
		print(f"d = {d}")

		kernel_class = PolynomialKernel(d)
		kernelized_matrix = kernel_class.apply_kernel(X_data)

		# q2) 5 folds CV:
		vali_corrects = []
		for fold in range(0, 5):
			print(f"fold: {fold + 1}")

			# Get indicies for training set and validation set:
			train_fold_index, vali_fold_index = five_fold_split(train_indices, fold)

			# Data sets:
			X_train = X_data[train_fold_index]
			Y_train = Y_data[train_fold_index]
			X_val = X_data[vali_fold_index]
			Y_val = Y_data[vali_fold_index]

			# initiate perceptron model class
			alphas = np.zeros((N_CLASSIFIERS, len(train_fold_index)))
			ovo_model = Perceptron(alphas, kernelized_matrix, N_CLASSES, method='ovo', classifier_values=classifier_values)

			for epoch in range(1, N_EPOCHS + 1):
				alphas = ovo_model.fit(train_fold_index, X_train, Y_train)

			# VALIDATION
			kernel_vals = kernelized_matrix[train_fold_index][:, vali_fold_index]
			pred = ovo_model.predict_ovo(alphas, kernel_vals)
			vali_correct = len(np.where((Y_val - pred) == 0)[0])

			vali_corrects.append(vali_correct)

		avg_validation_correct.append(np.sum(vali_corrects) / 5)

	# get d_star
	d_star = np.argmax(avg_validation_correct)

	# Retrain with d_star
	kernel_class = PolynomialKernel(d_star)
	kernelized_matrix = kernel_class.apply_kernel(X_data)

	# re-initiate the perceptron model class
	alphas = np.zeros((N_CLASSIFIERS, len(train_indices)))
	ovo_model = Perceptron(alphas, kernelized_matrix, N_CLASSES, method='ovo', classifier_values=classifier_values)

	for epoch in range(1, N_EPOCHS + 1):
		alphas = ovo_model.fit(train_indices, X_train, Y_train)

	# TEST
	X_test = X_data[test_indices]
	Y_test = Y_data[test_indices]

	kernel_vals = kernelized_matrix[train_indices][:, test_indices]
	pred = ovo_model.predict_ovo(alphas, kernel_vals)
	test_correct = len(np.where((Y_test - pred) == 0)[0])

	test_error = (len(test_indices) - test_correct) / len(test_indices)
	testing_errors.append(test_error)
	d_stars.append(d_star)

# Question 2
with open('../plots/part1/q6_2/log.txt', 'w') as f:
	f.write(f"test error per run: {testing_errors}\n")
	f.write(f"d_star per run: {d_stars}\n")
	f.write(f"mean +- std of testing error: {np.mean(testing_errors)} +- {np.std(testing_errors)}\n")
	f.write(f"mean +- std of d*: {np.mean(d_stars)} +- {np.std(d_stars)}\n")
	f.close()
