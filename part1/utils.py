import numpy as np
import matplotlib.pyplot as plt


def get_data(data_file):
	data = np.loadtxt(data_file)
	X = data[:, 1:]
	y = data[:, 0].astype(np.int)

	return X, y


def split_indices_with_shuffle(dataset, ratio=0.8):
	"""
	Train test split of indices
	Shuffle, then returns splitted indices
	"""
	shuffled_indices = np.random.permutation(len(dataset))
	train_size = round(len(shuffled_indices) * ratio)

	train_indices = shuffled_indices[:train_size]
	test_indices = shuffled_indices[train_size: len(shuffled_indices)]

	return train_indices, test_indices


def plot_digit(number_array: np.ndarray, y_true: str, save_path: str):
	number_matrix = np.reshape(number_array, (16, 16))
	plt.imshow(number_matrix)
	plt.axis('off')
	plt.title(f"y_true: {y_true}")
	plt.savefig(save_path)


def plot_accuracy(kernel_param, run, n_epochs, train_accuracy_all, test_accuracy_all, save_dir):
	fig1, ax1 = plt.subplots()
	ax1.plot(range(1, n_epochs), train_accuracy_all, label='train acc')
	ax1.plot(range(1, n_epochs), test_accuracy_all, label='test acc')
	ax1.legend()
	plt.xlabel('Epochs')
	plt.ylabel('accuracy')
	plt.savefig(f"{save_dir}/para{kernel_param}_run{run}.png")


def five_fold_split(train_indices, fold):
	size = int(len(train_indices) / 5)
	vali_fold_index = train_indices[fold * size:(fold + 1) * size]

	if fold == 0:
		train_fold_index = train_indices[(fold + 1) * size:]
	elif fold == 4:
		train_fold_index = train_indices[:fold * size]
	else:
		indices1 = train_indices[:fold * size]
		indices2 = train_indices[(fold + 1) * size:]
		train_fold_index = np.concatenate((indices1, indices2), axis=0)

	return train_fold_index, vali_fold_index
