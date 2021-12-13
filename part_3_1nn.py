import numpy as np
import part_3_perceptron as perceptron
import part_3_winnow as winnow

np.random.seed(0)

def predict(sample, dataset):
    neighbour = nearest_neighbour(sample, dataset)
    label = neighbour[-1]
    return label

# def update_2(sampel, weights):

def calc_distance(x1, x2):
    dist = 0
    for i in range(len(x1)):
        dist += np.square(x1[i] - x2[i])
    return np.sqrt(dist)

def nearest_neighbour(sample, dataset):
    distances = []
    for point in dataset:
        temp_dist = calc_distance(sample[:-1], point[:-1])
        distances.append((point, temp_dist))
    # np_distances = np.array(distances, dtype=object)
    # print(distances)
    # print(np_distances)
    distances.sort(key=lambda x:x[1])
    # print(distances)
    # np.sort(np_distances, key=lambda x:x[1])
    # print(np_distances)
    return distances[0][0]

def train_weights(training_set, n):
    weights = np.ones(len(training_set[0]))
    mistakes = 0
    for t in range (len(training_set[0])):
        for sample in training_set:
            y_hat_t = predict(sample, weights)
            print(y_hat_t)
            y_t = sample[-1]
            if y_t != y_hat_t:
                # weights = update(y_t, sample, n, weights)
                mistakes += 1
        # print(weights)
    return weights

if __name__ == '__main__':
    n = 4
    training_samples = 10
    testing_samples = 1000

    training_data = perceptron.random_sample(n, training_samples)
    testing_data = perceptron.random_sample(n, testing_samples)

    training_labels = perceptron.label(training_data)
    testing_labels = perceptron.label(testing_data)

    training_dataset = perceptron.create_dataset(training_data, training_labels)
    testing_dataset = perceptron.create_dataset(testing_data, testing_labels)
    print(testing_dataset[1][:-1])
    # distance = []
    # for i in range(len(training_dataset[:-1])-1):
    #     for x in range(len(training_dataset[:-1])-1):
    #         distance.append(calc_distance(training_dataset[i], training_dataset[x]))
    # print(max(distance))

    closest_neighbour = nearest_neighbour(testing_dataset[1], training_dataset)
    print(closest_neighbour)
    print(calc_distance(closest_neighbour[:-1], testing_dataset[1][:-1]))

    mistakes = 0
    for i in range(len(testing_dataset)):
        prediction = predict(testing_dataset[i], training_dataset)
        if prediction != testing_dataset[i][-1]:
            mistakes += 1
    print(mistakes/len(testing_dataset))
