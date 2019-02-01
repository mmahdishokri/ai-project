import idx2numpy
import numpy as np
from train import *


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
    # I changed it into list
    # todo manage it!

    def put_random_weights(self):
        size = self.weights.__sizeof__()
        # self.weights = []
        for i in range(size):
            self.weights = \
                np.append(self.weights, np.random.random_sample())

    def __init__(self, num):
        self.num = num
        self.put_random_weights()


# we have Layer(s) in Neural Network which contains some Cells
class Layer:
    cells = []
    for i in range(10):
        cells.append(Cell(i))


def announce_error():
    #   todo the sum of the Squares of Differs
    return Cell(1)


def start_training_the_layer(layer):
    train_images, train_labels = get_training_data()
    for i in range(train_images.size):
        image = Image()
        image.pixels = train_images[i]
        image.num = train_labels[i]
        train_layer(layer, image)


start_training_the_layer(Layer())
