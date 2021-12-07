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
    # print(a.shape)
    # print(b.shape)
    # b = (328, 257)
    return np.dot(a, b) ** 3
    # return np.matmul(a, b) ** 3


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
        sum += cl[i] * kerval(pat, dat[i][1:])
    return sum

def trainGen(data):
    mistakes = 0
    # print(data)
    GLBcls = np.zeros((3, len(data)))
    for i in range(len(data)):
        val = data[i][0]

        preds = []
        # GLBcls = np.zeros((3, len(data)))
        for j in range(3):
            preds.append(classPredK(data, data[i][1:], GLBcls[j]))

        maxc = -10000000000000000
        maxi = 0

        for z in range(3):
            if val == z:
                y = 1
            else:
                y = -1
            if (y * preds[z] <= 0):
                GLBcls[z, i] = GLBcls[z, i] - mysign(preds[z])
            if preds[z] > maxc:
                maxc = preds[z]
                maxi = z
            # print(maxc)
        if maxi != val:
            mistakes += 1
    return mistakes

def testClassifiers(data, test_data):
    mistakes = 0
    GLBcls = np.zeros((3, len(data)))
    for i in range(len(test_data)):
        val = test_data[i][0]
        preds = []
        for j in range(3):
            preds.append(classPredK(data, test_data[i][1:], GLBcls[j]))

        maxc = -10000000000000000
        maxi = 0

        for z in range(3):
            # i dont understand the point of this if statement
            if val == z:
                y = 1
            else:
                y = -1
            if preds[z] > maxc:
                maxc = preds[z]
                maxi = z
            # print(maxc)
        if maxi != val:
            mistakes += 1
    print(mistakes)
    return mistakes/len(test_data)

def demo(train, test):
    i = 0
    rtn = []
    GLBcls = np.zeros((3, len(train)))
    for i in range(3):
        rtn = trainGen(train)
        print(f"Training - epoch {str(i)} required {str(rtn)} with {str(rtn)} mistakes out of {str(len(train))} items.\n")
        rtn = testClassifiers(train, test)
        print(f"Testing - epoch  {str(i)} required {str(rtn)} with a test error of \n")

if __name__ == '__main__':

    training_data = get_data('dtrain123.dat')
    testing_data = get_data('dtest123.dat')
    # print(training_data.shape)
    # print(test_data.shape)
    # plotChar(training_data[3])
    # train_mistakes = trainGen(training_data)
    # print(train_mistakes)
    # test_mistakes = testClassifiers(training_data, testing_data)
    # print(test_mistakes)
    demo(training_data, testing_data)

