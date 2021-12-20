import numpy as np
from abc import ABC, abstractmethod


class Kernel(ABC):
	def __init__(self, param):
		self.param = param

	@abstractmethod
	def apply_kernel(self, X):
		pass


class PolynomialKernel(Kernel):

	def apply_kernel(self, X):
		return np.dot(X, X.T) ** self.param


class GaussianKernel(Kernel):

	def apply_kernel(self, X):
		xi_squared = np.sum(X ** 2, axis=1).reshape((X.shape[0], 1))
		xj_squared = np.sum(X ** 2, axis=1).reshape((1, X.shape[0]))

		complete_squared = xi_squared + xj_squared - 2 * np.dot(X, X.T)

		return np.exp(-self.param * complete_squared)

