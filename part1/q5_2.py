""".
Repeat 1 and 2 (d∗ is now c and {1,...,7} is now S) above with a Gaussian kernel c the width of the kernel is now
a parameter which must be optimised during cross-validation however, you will also need to perform some initial
experiments to a decide a reasonable set S of values to cross- validate c over.
"""
import numpy as np
from tqdm import tqdm
from utils import get_data, five_fold_split, split_indices_with_shuffle
from kernels import GaussianKernel
from perceptrons import Perceptron


X_data, Y_data = get_data("../zipcombo.dat")

N_CLASSES = 10
N_EPOCHS = 5
RUNS = 20
# C list is decided after the initial experiments from q5_1.py
C = [0.005, 0.006, 0.007, 0.008, 0.009, 0.01]

testing_errors = []
c_stars = []

# Loop for 20 runs
for run in tqdm(range(1, RUNS + 1)):
	print(f"Run: {run}")

	# split indices with shuffling
	train_indices, test_indices = split_indices_with_shuffle(Y_data)

	avg_validation_correct = []

	for c in C:
		print(f"c = {c}")

		kernel_class = GaussianKernel(c)
		kernelized_matrix = kernel_class.apply_kernel(X_data)

		# 5 folds CV:
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
			alphas = np.zeros((N_CLASSES, len(train_fold_index)))
			model = Perceptron(alphas, kernelized_matrix, N_CLASSES)

			for epoch in range(1, N_EPOCHS + 1):
				alphas = model.fit(train_fold_index, X_train, Y_train)

			# VALIDATION
			kernel_vals = kernelized_matrix[train_fold_index][:, vali_fold_index]
			pred = np.array(list(map(np.argmax, zip(*(alphas.dot(kernel_vals))))))
			vali_correct = len(np.where((Y_val - pred) == 0)[0])

			vali_corrects.append(vali_correct)

		avg_validation_correct.append(np.sum(vali_corrects) / 5)

	# get c_star
	c_star = C[np.argmax(avg_validation_correct)]
	print(f"c_star: {c_star}")

	# Retrain with c_star
	kernel_class = GaussianKernel(c_star)
	kernelized_matrix = kernel_class.apply_kernel(X_data)

	# re-initiate the perceptron model class
	alphas = np.zeros((N_CLASSES, len(train_indices)))
	model = Perceptron(alphas, kernelized_matrix, N_CLASSES)

	for epoch in range(1, N_EPOCHS + 1):
		alphas = model.fit(train_indices, X_train, Y_train)

	# TEST
	X_test = X_data[test_indices]
	Y_test = Y_data[test_indices]

	kernel_vals = kernelized_matrix[train_indices][:, test_indices]
	pred = np.array(list(map(np.argmax, zip(*(alphas.dot(kernel_vals))))))
	test_correct = len(np.where((Y_test - pred) == 0)[0])

	test_error = (len(test_indices) - test_correct) / len(test_indices)
	testing_errors.append(test_error)
	c_stars.append(c_star)


with open('../plots/part1/q5_2/log.txt', 'w') as f:
	f.write(f"test error per run: {testing_errors}\n")
	f.write(f"c_star per run: {c_stars}\n")
	f.write(f"mean +- std of testing error: {np.mean(testing_errors)} +- {np.std(testing_errors)}\n")
	f.write(f"mean +- std of c_star: {np.mean(c_stars)} +- {np.std(c_stars)}\n")
	f.close()
