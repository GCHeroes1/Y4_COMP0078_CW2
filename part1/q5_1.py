""".
Repeat 1 and 2 (d∗ is now c and {1,...,7} is now S) above with a Gaussian kernel c the width of the kernel is now
a parameter which must be optimised during cross-validation however, you will also need to perform some initial
experiments to a decide a reasonable set S of values to cross- validate c over.
"""
import numpy as np
from tqdm import tqdm
from utils import get_data, split_indices_with_shuffle, plot_accuracy
from kernels import GaussianKernel
from perceptrons import Perceptron

N_CLASSES = 10
N_EPOCHS = 5
RUNS = 20
PLOT_ACCURACY = True
BASE_DIR = '../plots/part1/q5_1/trial3'

# "For an initial experiment for the parameter of gaussian kernel,
# a logarithmic grid with basis 10 is often helpful.
# Using a basis of 2, a finer tuning can be achieved but at a much higher cost."
# source of the above sentence: https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html

"""
Following trials to search an optimal parameter of 'c' in gaussian kernel
is decided based on the train and test accuracy graphs. 
We will see if the c value is in a reasonable range
"""
# Trial 1
C = np.logspace(-3, 4, 5)  # [0.001, 0.05623413251903491, 3.1622776601683795, 177.82794100389228, 10000.0]
'''
From trial 1 we could see that the model starts to overfit on data from c=0.05623413251903491, 
and when c=177 and c=10000, the train accuracy fluctuates heavily.
'''
# Trial 2
C = np.logspace(-3, 0, 5)  # [0.001, 0.005623413251903491, 0.03162277660168379, 0.1778279410038923, 1.0]
'''
Therefore, we changed the range like above.
Now that we found an approximate upperbound of c value,
we now want to see how the model behaves when c is less than 0.001
'''
# Trial 3
C = [0.0005, 0.001, 0.005, 0.007, 0.008, 0.009]
'''
In the c value that's less than 0.001, we see that test accuracy is higher than the train accuracy.
The model is optimised on training data but accuracy comes higher is testing data 
'''

# get data as X and Y
X_data, Y_data = get_data("../zipcombo.dat")

train_errors_mean = []
train_errors_std = []
test_errors_mean = []
test_errors_std = []

# iterate through all c values
for c in tqdm(C):
	print(f"c = {c}")

	train_error_per_run = []
	test_error_per_run = []

	# kernelize data
	kernel_class = GaussianKernel(c)
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
			plot_accuracy(c, run, N_EPOCHS + 1, train_accuracies, test_accuracies,
			              save_dir=BASE_DIR)

	# mean of errors and std:
	train_errors_mean.append(np.mean(train_error_per_run))
	train_errors_std.append(np.std(train_error_per_run))
	test_errors_mean.append(np.mean(test_error_per_run))
	test_errors_std.append(np.std(test_error_per_run))

with open(f'{BASE_DIR}/log.txt', 'w') as f:
	f.write(f"Kernel parameter C: {C}\n\n")
	f.write(f"Mean train error: {train_errors_mean}\n")
	f.write(f"STD training error: {train_errors_std}\n")
	f.write(f"Mean testing error: {test_errors_mean}\n")
	f.write(f"STD testing error: {test_errors_std}\n")
	f.close()
