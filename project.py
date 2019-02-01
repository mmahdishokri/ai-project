import idx2numpy
import numpy as np


def get_training_data():
    train_images = idx2numpy.convert_from_file('data/train-images-idx3-ubyte')
    train_labels = idx2numpy.convert_from_file('data/train-labels-idx1-ubyte')
    return train_images, train_labels


def get_testing_data():
    testing_labels = idx2numpy.convert_from_file('data/t10k-labels-idx1-ubyte')
    testing_images = idx2numpy.convert_from_file('data/t10k-images-idx3-ubyte')
    return testing_images, testing_labels


class Image:
    pixels = np.array
    num = int


class Cell:
    output = np.float
    weights = np.array

    def put_random_weights(self):
        # size = self.weights.__sizeof__()
        size = 28 * 28
        # self.weights = []
        for i in range(size):
            self.weights = \
                np.append(self.weights, np.random.random_sample())
        self.weights = np.delete(self.weights, 0)

    def __init__(self, num):
        self.num = num
        self.put_random_weights()


# we have Layer(s) in Neural Network which contains some Cells
class Layer:
    def __init__(self):
        self.cells = []
        for i in range(10):
            self.cells.append(Cell(i))


def announce_error():
    # todo the sum of the Squares of Differs
    return Cell(1)
