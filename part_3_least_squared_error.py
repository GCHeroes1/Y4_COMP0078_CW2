import numpy as np
import part_3_perceptron as perceptron
import part_3_winnow as winnow

np.random.seed(0)

# least squares, find function f(x) = np.dot(w.T, x) which best fits data
# prediction is y_i = np.dot(w.T, x_i)

def MSE(sample):
    print(sample)
    LSE = 0
    for i in range(len(sample)):
        LSE += np.square(np.sum(sample[i][:-1]))
    print(LSE)

def predict(sample, weights):
    return np.dot(sample, weights.T)
    # if y_hat_t >= 0:
    #     return
    # else:
    #     return -1

# def update_2(sampel, weights):

def update(y, x, n, weight): #RSS -> residual sum of squares
    sum = 0
    for i in range(n):
        sum += np.square(y - np.dot(weight.T, x[i]))
    return sum / n

def train_weights(training_set, n):
    # deprecated??? not sure how to update weights for least squares, i dont think you do? i guess you'd update a bias term...
    weights = np.ones(len(training_set[0]))
    mistakes = 0
    for t in range (len(training_set[0])):
        for sample in training_set:
            y_hat_t = predict(sample, weights)
            print(y_hat_t)
            y_t = sample[-1]
            if y_t != y_hat_t:
                weights = update(y_t, sample, n, weights)
                print(sample)
                weight = np.linalg.pinv(sample)
                print(weight)
                # for i in range (len(weights)):
                # print("made a mistake")
                    # weights[i] = weights[i] * 2 ** np.dot((y_t - y_hat_t), sample[i])
                    # weights = weights + np.dot(y_t, sample)
                mistakes += 1
        # print(weights)
    return weights

if __name__ == '__main__':
    n = 100
    training_samples = 400
    testing_samples = 100

    training_data = perceptron.random_sample(n, training_samples)
    testing_data = perceptron.random_sample(n, testing_samples)

    training_labels = perceptron.label(training_data)
    testing_labels = perceptron.label(testing_data)

    training_dataset = perceptron.create_dataset(training_data, training_labels)
    weights = np.dot(np.linalg.pinv(training_data), training_labels)
    # print(weights[0])
    # print((weights * training_labels)[0])
    # print(weights)
    testing_dataset = perceptron.create_dataset(testing_data, testing_labels)
    mistakes = 0
    for i in range(len(testing_dataset)):
        prediction = np.dot(testing_dataset[i][:-1], weights)
        rounded_pred = np.round(prediction)
        # print(rounded_pred)
        # print(np.around(np.dot(testing_dataset[i][:-1], weights)), decimals=2)
        if rounded_pred != testing_dataset[i][-1]:
            mistakes += 1
    # print(np.round(np.dot(testing_dataset[i][:-1], weights)))
    # print(testing_dataset[i][-1])

    print(mistakes/len(testing_dataset))
