import numpy as np
import part_3_perceptron as perceptron
import part_3_winnow as winnow

np.random.seed(0)

def MSE(sample):
    print(sample)
    LSE = 0
    for i in range(len(sample)):
        LSE += np.square(np.sum(sample[i][:-1]))
    print(LSE)

def predict(sample, weights):
    return np.dot(sample, weights)
    # if y_hat_t >= 0:
    #     return
    # else:
    #     return -1

def update(y, x, n, weight):
    sum = 0
    for i in range(n):
        sum += np.square(y - np.dot(weight.T, x[i]))
    return sum

def train_weights(training_set, n):
    weights = np.ones(len(training_set[0]))
    mistakes = 0
    for t in range (len(training_set[0])):
        for sample in training_set:
            y_hat_t = predict(sample, weights)
            y_t = sample[-1]
            if y_t != y_hat_t:
                weights = update(y_t, sample, n, weights)
                # for i in range (len(weights)):
                print("made a mistake")
                    # weights[i] = weights[i] * 2 ** np.dot((y_t - y_hat_t), sample[i])
                    # weights = weights + np.dot(y_t, sample)
                mistakes += 1
        # print(weights)
    return weights

if __name__ == '__main__':
    n = 5
    training_samples = 100
    testing_samples = 1000

    training_data = perceptron.random_sample(n, training_samples)*0.5
    testing_data = perceptron.random_sample(n, testing_samples)

    training_labels = perceptron.label(training_data)
    testing_labels = perceptron.label(testing_data)

    training_dataset = perceptron.create_dataset(training_data, training_labels)
    testing_dataset = perceptron.create_dataset(testing_data, testing_labels)

    weights = train_weights(training_dataset, n)
    print(weights)
    # prediction = predict(training_dataset[0], weights=[1, 1, 1])
    # print("works")
    # print(prediction)

