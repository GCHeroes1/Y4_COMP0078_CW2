import numpy as np
import matplotlib.pyplot as plt


class PolynomialRegression:

    def __init__(self, degree=None, need_transform=True):
        self.degree = degree
        self.need_transform = need_transform
        if need_transform:
            assert degree is not None
        self.X = None
        self.Y = None
        self.m = None
        self.n = None
        self.weights = None

    # function to transform X
    def transform(self, X):
        self.m, self.n = X.shape
        # initialize X_transform
        X_transform = np.ones((self.m, 1))

        for j in range(self.degree + 1):
            if j != 0:
                x_pow = np.power(X, j)
                # append x_pow to X_transform 2-D array
                X_transform = np.append(X_transform, x_pow.reshape(-1, 1), axis=1)

        return X_transform

    # model training
    def fit(self, X, Y):
        self.X = X
        self.Y = Y

        if self.need_transform:
            # transform X for polynomial h( x ) = w0 * x^0 + w1 * x^1 + w2 * x^2 + ........+ wn * x^n
            X_transform = self.transform(self.X)
            self.weights = np.linalg.inv(X_transform.T.dot(X_transform)).dot(X_transform.T).dot(self.Y)
        else:
            self.weights = np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T).dot(self.Y)

        return self

    def predict(self, X):
        if self.need_transform:
            # transform X for polynomial h( x ) = w0 * x^0 + w1 * x^1 + w2 * x^2 + ........+ wn * x^n
            X_transform = self.transform(X)
            return np.dot(X_transform, self.weights)
        else:
            return np.dot(X, self.weights)


if __name__ == "__main__":
    X = np.array([[1, 2, 3, 4]])
    X = X.T
    Y = np.array([3, 2, 0, 5])

    # model training
    model = PolynomialRegression(degree=3)

    model.fit(X, Y)
    print(model.weights)

    # Prediction on training set
    Y_pred = model.predict(X)

    # Visualization
    plt.scatter(X, Y, color='blue')
    plt.plot(X, Y_pred, color='orange')
    plt.title('X vs Y')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
