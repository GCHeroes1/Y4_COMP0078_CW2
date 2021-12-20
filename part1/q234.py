"""
2.
Cross-validation: Perform 20 runs : when using the 80% training data split from within to perform 5-fold
cross-validation to select the “best” parameter d∗ then retrain on full 80% training set using d∗ and then record the
test errors on the remaining 20%. Thus you will find 20 d∗ and 20 test errors. Your final result will consist of a
mean test error±std and a mean d∗ with std.

3.
Confusion matrix: Perform 20 runs : when using the 80% training data split that further to perform 5-fold
cross-validation to select the “best” parameter d∗
retrain on the full “80%” training set using d∗ and then produce a confusion matrix.
Here the goal is to find “confusions” thus if the true label (on the test set) was "7" and "2" was
predicted then a “error” should recorded for “(7,2)”; the final output will be a 10 × 10 matrix where each cell
contains a confusion error rate and its standard deviation (here you will have averaged over the 20 runs). Note the
diagonal will be 0.

4.
Within the dataset relative to your experiments there will be five hardest to predict correctly “pixelated images.”
Print out the visualisation of these five digits along with their labels.
Is it surprising that these are hard to predict?
"""
import numpy as np
from tqdm import tqdm
from utils import get_data, plot_accuracy, five_fold_split, plot_digit, split_indices_with_shuffle
from kernels import PolynomialKernel
from perceptrons import Perceptron
import seaborn as sns

X_data, Y_data = get_data("../zipcombo.dat")

N_CLASSES = 10
N_EPOCHS = 2
RUNS = 20

# q2
testing_errors = []
d_stars = []

# q3
confusion_matrix = []

# q4
pred_mistakes = np.zeros((Y_data.shape[0], 1))

# q2
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

			# Get indicies for training set and validation set
			train_fold_index, vali_fold_index = five_fold_split(train_indices, fold)

			# Data sets
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

	# get d_star
	d_star = np.argmax(avg_validation_correct)

	# Retrain with d_star
	kernel_class = PolynomialKernel(d_star)
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
	d_stars.append(d_star)

	# q3: Confusion Matrix construction
	confusion = np.zeros((10, 10))
	count = np.zeros((10, 1))
	for true, hat in zip(Y_test, pred):
		count[true] += 1
		if true != hat:
			confusion[true, hat] += 1
	confusion = confusion / count

	confusion_matrix.append(confusion)

	# q4) 5 hardest digits
	pred = np.array(list(map(np.argmax, zip(*(alphas.dot(kernelized_matrix[train_indices]))))))
	for i in range(Y_data.shape[0]):
		if not pred[i] == Y_data[i]:
			pred_mistakes[i] += 1

worst_5_indices = np.argsort(-pred_mistakes.T)[0, :5]

# Question 2
with open('../plots/part1/q2/log.txt', 'w') as f:
	f.write(f"test error per run: {testing_errors}\n")
	f.write(f"d_star per run: {d_stars}\n")
	f.write(f"mean +- std of testing error: {np.mean(testing_errors)} +- {np.std(testing_errors)}\n")
	f.write(f"mean +- std of d*: {np.mean(d_stars)} +- {np.std(d_stars)}\n")
	f.close()


# Question 3
def flatten(data):
	flattend = []
	for x in data:
		if hasattr(x, '__iter__'):
			flattend.extend(flatten(x))
		else:
			flattend.append(x)
	return flattend


final_mean_matrix = np.mean(confusion_matrix, axis=0)
final_std_matrix = np.std(confusion_matrix, axis=0)

flattened_mean = flatten(final_mean_matrix)
flattened_std = flatten(final_std_matrix)

labels = [f"{round(v1, 4)}\n±{round(v2, 3)}" for v1, v2 in zip(flattened_mean, flattened_std)]
labels = np.asarray(labels).reshape(10, 10)

cf_plot = sns.heatmap(final_mean_matrix, annot=labels, fmt='', annot_kws={"size": 6})
fig = cf_plot.get_figure()
fig.suptitle('Confusion Matrix')
fig.savefig('../plots/part1/q3/cf_matrix.png')

# Question 4
for i, idx in enumerate(worst_5_indices):
	plot_digit(X_data[idx], str(Y_data[idx]), save_path=f'../plots/part1/q4/idx_{i}_{idx}.png')
