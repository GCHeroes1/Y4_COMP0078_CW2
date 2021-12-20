import numpy as np
from tqdm import tqdm
from utils import get_data, plot_accuracy
from kernels import PolynomialKernel
from perceptrons import Perceptron

# global training_data
# global testing_data
#
#
# def mysign(x):
# 	if x <= 0:
# 		return -1
# 	return 1
#
#
# def clearGLBcls(data):  # todo: wtf is this function, creates a big boi array?
# 	return np.zeros((3, len(data)))
#
#
# def classPredK(dat, pat, cl):  # Compute prediction of classifier on a particular pattern
# 	index = len(cl)
# 	sum = 0
# 	for i in range(index):
# 		sum += cl[i] * kerval(pat, dat[i][1:])
# 	return sum
#
#
# def trainGen(data):
# 	mistakes = 0
# 	# print(data)
# 	GLBcls = np.zeros((3, len(data)))
# 	for i in range(len(data)):
# 		val = data[i][0]
#
# 		preds = []
# 		# GLBcls = np.zeros((3, len(data)))
# 		for j in range(3):
# 			preds.append(classPredK(data, data[i][1:], GLBcls[j]))
#
# 		maxc = -10000000000000000
# 		maxi = 0
#
# 		for z in range(3):
# 			if val == z:
# 				y = 1
# 			else:
# 				y = -1
# 			if y * preds[z] <= 0:
# 				GLBcls[z, i] = GLBcls[z, i] - mysign(preds[z])
# 			if preds[z] > maxc:
# 				maxc = preds[z]
# 				maxi = z
# 		# print(maxc)
# 		if maxi != val:
# 			mistakes += 1
# 	return mistakes
#
#
# def testClassifiers(data, test_data):
# 	mistakes = 0
# 	GLBcls = np.zeros((3, len(data)))
# 	for i in range(len(test_data)):
# 		val = test_data[i][0]
# 		preds = []
# 		for j in range(3):
# 			preds.append(classPredK(data, test_data[i][1:], GLBcls[j]))
#
# 		maxc = -10000000000000000
# 		maxi = 0
#
# 		for z in range(3):
# 			# i dont understand the point of this if statement
# 			if val == z:
# 				y = 1
# 			else:
# 				y = -1
# 			if preds[z] > maxc:
# 				maxc = preds[z]
# 				maxi = z
# 		# print(maxc)
# 		if maxi != val:
# 			mistakes += 1
# 	print(mistakes)
# 	return mistakes / len(test_data)
#
#
# def demo(train, test):
# 	i = 0
# 	rtn = []
# 	GLBcls = np.zeros((3, len(train)))
# 	for i in range(3):
# 		rtn = trainGen(train)
# 		print(
# 			f"Training - epoch {str(i)} required {str(rtn)} with {str(rtn)} mistakes out of {str(len(train))} items.\n")
# 		rtn = testClassifiers(train, test)
# 		print(f"Testing - epoch  {str(i)} required {str(rtn)} with a test error of \n")


# -------
X_dataset, Y_dataset = get_data("../zipcombo.dat")

# K-Class Classifier Parameters:
Ker_Meth = "Poly"
n_classes = 10

# Question 1a):
Training_Error_Mean = []
Training_Error_Stdev = []
Testing_Error_Mean = []
Testing_Error_Stdev = []

# ---------------Training--------------------------
# Question 1a):
# Loop for d from 1 to 7: range(1, 8)
for Ker_para in tqdm(range(1, 8)):
	print("d = {}".format(Ker_para))

	# Question 1a:
	Training_Error_per_run = []
	Testing_Error_per_run = []

	# Precalculate Kernel
	KF = PolynomialKernel(Ker_para)
	KN = KF.apply_kernel(X_dataset)  # Kernel matrix

	# 20 runs: range(1, 21)
	for run in tqdm(range(1, 21)):
		print("Run: {}".format(run))

		# Index Shuffle for fitting:
		ran_sample_idx = np.random.permutation(len(Y_dataset))
		split_pt = round(len(ran_sample_idx) * 0.8)

		# Index for Debug:
		# ran_sample_idx       = range(len(Y_dataset))
		# split_pt             = round(len(ran_sample_idx)*0.41911)

		ran_train_sample_idx = ran_sample_idx[:split_pt]
		ran_test_sample_idx = ran_sample_idx[split_pt:len(ran_sample_idx)]

		# Data sets:
		X_train = X_dataset[ran_train_sample_idx]
		Y_train = Y_dataset[ran_train_sample_idx]
		X_test = X_dataset[ran_test_sample_idx]
		Y_test = Y_dataset[ran_test_sample_idx]

		nsamples = len(ran_train_sample_idx)
		n_test_samples = len(ran_test_sample_idx)

		# Initialize:
		alphas = np.zeros((n_classes, nsamples))
		# alphas               = np.zeros((n_classes, nsamples + n_test_samples))

		# Start Model:
		ovrop = Perceptron(alphas, KN, n_classes)

		# Results memory:
		Train_correct_All = []
		Train_accuracy_All = []
		Train_mistakes_All = []
		Test_correct_All = []
		Test_accuracy_All = []
		Test_mistakes_All = []

		n_epochs = 20
		for epoch in range(1, n_epochs + 1):
			alphas = ovrop.fit(ran_train_sample_idx, X_train, Y_train)

			# ---------------Training Accuracy-----------------
			# Train_correct = 0

			# kernel_vals   = KN[:, ran_train_sample_idx]
			kernel_vals = KN[ran_train_sample_idx]
			kernel_vals = kernel_vals[:, ran_train_sample_idx]
			pred = np.array(list(map(np.argmax, zip(*(alphas.dot(kernel_vals))))))
			Train_correct = len(np.where((Y_train - pred) == 0)[0])

			# Record results:
			Train_correct_All.append(Train_correct)

			Train_accuracy = Train_correct / nsamples
			Train_accuracy_All.append(Train_accuracy)

			Train_mistakes = nsamples - Train_correct
			Train_mistakes_All.append(Train_mistakes)

			# print("\tTraining correctness = ", Train_correct)
			# print("\tTraining accuracy    = ", Train_accuracy)
			# print("\tTraining mistake(s)  = ", Train_mistakes, "\n")

			# ---------------Testing Accuracy-------------------
			# print(">>Testing:")
			# Test_correct = 0

			# kernel_vals  = KN[:, ran_test_sample_idx]
			kernel_vals = KN[ran_train_sample_idx]
			kernel_vals = kernel_vals[:, ran_test_sample_idx]
			pred = np.array(list(map(np.argmax, zip(*(alphas.dot(kernel_vals))))))
			Test_correct = len(np.where((Y_test - pred) == 0)[0])

			# Record results:
			Test_correct_All.append(Test_correct)

			Test_accuracy = Test_correct / n_test_samples
			Test_accuracy_All.append(Test_accuracy)

			Test_mistakes = n_test_samples - Test_correct
			Test_mistakes_All.append(Test_mistakes)

			# print("\tTesting correctness  = ", Test_correct)
			# print("\tTesting accuracy     = ", Test_accuracy)
			# print("\tTesting mistake(s)   = ", Test_mistakes, "\n")
			# print(f"{Train_accuracy:0.4f} and Test accuracy {Test_accuracy:0.4f}.")
			epoch += 1

		# Question 1a): Save error rate:
		train_error_per_run = Train_mistakes / nsamples
		test_error_per_run = Test_mistakes / n_test_samples
		print(f"Train error per run: {train_error_per_run}")
		print(f"Test error per run: {test_error_per_run}")
		Training_Error_per_run.append(train_error_per_run)
		Testing_Error_per_run.append(test_error_per_run)

		# Plot accuracy graph
		plot_accuracy(Ker_para, run, n_epochs + 1, Train_accuracy_All, Test_accuracy_All)

	# Question 1a): Error mean and std:
	Training_Error_Mean.append(np.mean(Training_Error_per_run))
	Training_Error_Stdev.append(np.std(Training_Error_per_run))
	Testing_Error_Mean.append(np.mean(Testing_Error_per_run))
	Testing_Error_Stdev.append(np.std(Testing_Error_per_run))

with open('../plots/part1/q1/log.txt', 'w') as f:
	f.write(f"Mean train error: {Training_Error_Mean}\n")
	f.write(f"STD training error: {Training_Error_Stdev}\n")
	f.write(f"Mean testing error: {Testing_Error_Mean}\n")
	f.write(f"STD testing error: {Testing_Error_Stdev}\n")
	f.close()
