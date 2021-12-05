import numpy as np
import matplotlib.pyplot as plt

def get_data(dat_file):
    return np.loadtxt(dat_file)

def plotChar(number_array):
    number = number_array[1:]
    number_matrix = np.reshape(number, (16, 16))
    # plt.title(f"The number {str(number_array[0])[0]}")
    # plt.pcolormesh(number_matrix)
    # plt.show()

def kerval(a, b):    # todo: figure out what a and b are here
    # a = (328, 257)
    # b = (328, 257)
    return np.dot(a, b) ^ 3

def mysign(x):
    if x <= 0:
        return -1
    return 1

def clearGLBcls(data):                   # todo: wtf is this function, creates a big boi array?
    return np.zeros((3, len(data)))

def classPredK(dat, pat, cl):   # Compute prediction of classifier on a particular pattern
    index = len(cl)
    sum = 0
    for i in range(index):
        sum += cl[i] * kerval(pat, dat[1:])
    return sum

def trainGen(data):
    for i in range(len(data)):
        val = data[i][0]
        print(val)
        # table is called with these parameters
        # expr = classpredk[dat, Take[dat[[i]], {2, 257}]
        # i = GLBcls[[j]]
        # j = {j, 1, 3}
        predictions = 0
        GLBcls = np.zeros((3, len(data)))
        for j in range(3):
            predictions += classPredK(data, data[1:], GLBcls[j])
        print(predictions)

        # maxc = -10000000000000000 #ineficiency?
        #
        # for j in range(3):
        #     if val == j:
        #         y = 1
        #     else:
        #         y = -1
        #     if (y * predictions[j] <= 0)
if __name__ == '__main__':

    training_data = get_data('dtrain123.dat')
    test_data = get_data('dtest123.dat')
    # print(training_data.shape)
    # print(test_data.shape)
    plotChar(training_data[1])
    trainGen(training_data)

