import idx2numpy
import numpy as np
from train import *


train_images = idx2numpy.convert_from_file('data/train-images-idx3-ubyte')
train_labels = idx2numpy.convert_from_file('data/train-labels-idx1-ubyte')
testing_images = idx2numpy.convert_from_file('data/t10k-images-idx3-ubyte')
testing_labels = idx2numpy.convert_from_file('data/t10k-labels-idx1-ubyte')


class Image:
    pixels = np.array
    num = int


class Cell:
    def __init__(self, num):
        self.num = num
    weights = np.array
    output = np.float


# we have Layer(s) in Neural Network which contains some Cells
class Layer:
    cells = []
    for i in range(10):
        cells.append(Cell(i))


def announce_output(layer):
    #   todo print the max
    return Cell(1)


def announce_error():
    #   todo the sum of the Squares of Differs
    return Cell(1)


def put_random_weights(cell):
    size = cell.weights.size
    cell.weights = []
    for i in range(size):
        cell.weights.append(np.random.random_sample())


def start_training_the_layer(layer):
    for i in range(train_images.size):
        image = Image()
        image.pixels = train_images[i]
        image.num = train_labels[i]
        train_layer(layer, image)


start_training_the_layer(Layer())
