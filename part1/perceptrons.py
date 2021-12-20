import numpy as np


class Perceptron:
	"""
	Unified Class for both
	One Vs Rest, and One Vs One
	"""
	def __init__(self, alphas, kernel_matrix, n_classes, method='ovr', classifier_values=None):
		self.alphas = alphas
		self.kernel_matrix = kernel_matrix
		self.train_indices = None
		self.method = method

		if method == 'ovr':
			self.n_classifiers = n_classes
		elif method == 'ovo':
			self.n_classifiers = int(n_classes * (n_classes - 1) / 2)

		self.classifier_values = classifier_values

	@staticmethod
	def _sign(val):
		ret = np.where(val <= 0.0, -1, 1)
		return ret

	def fit(self, train_indices, X, y):
		self.train_indices = train_indices

		for i in range(X.shape[0]):
			cls_vals = np.zeros((self.n_classifiers,))
			kernel_values = self.kernel_matrix[self.train_indices, self.train_indices[i]]
			cls_vals += self.alphas.dot(kernel_values)

			y_hat = self._sign(cls_vals)
			y_true = self._encode_label(y[i])

			if self.method == 'ovr':
				condition = np.multiply(cls_vals, y_true) <= 0
				self.alphas[:, i] -= np.where(condition, y_hat, 0)
			elif self.method == 'ovo':
				condition = y_hat != y_true
				self.alphas[:, i] += np.where(condition, y_true, 0)

		return self.alphas

	def predict_ovo(self, alphas, kernel_vals):
		"""
		Only for prediction of one vs one
		"""
		y_pred = (alphas.dot(kernel_vals) > 0) * 2 - 1
		vote = np.zeros((10, kernel_vals.shape[1]))
		for cls_i in range(self.classifier_values.shape[0]):
			row_pos = int(self.classifier_values[cls_i, 0])
			row_neg = int(self.classifier_values[cls_i, 1])

			pos = y_pred[cls_i, :] == 1
			neg = y_pred[cls_i, :] == -1

			vote[row_pos, pos] += 1
			vote[row_neg, neg] += 1

		return np.argmax(vote, axis=0)

	def _encode_label(self, label):
		if self.method == 'ovr':
			labels = np.full((self.n_classifiers,), -1)
			labels[label] = 1
			return labels
		elif self.method == 'ovo':
			labels = np.zeros((self.n_classifiers,))
			for i in range(self.n_classifiers):
				if self.classifier_values[i][0] == label:
					labels[i] = 1
				elif self.classifier_values[i][1] == label:
					labels[i] = -1
			return labels

