import idx2numpy
import numpy as np

# First to read the training dataSet
images = idx2numpy.convert_from_file('train-images-idx3-ubyte.idx')
labels = idx2numpy.convert_from_file('train-labels-idx1-ubyte.idx')
images_read = open('train-images-idx1-ubyte.idx', 'rb')
labels_read = open('train-labels-idx1-ubyte.idx', 'rb')

# to write on it
f_write = open('myfile_copy2.idx', 'wb')
idx2numpy.convert_to_file(f_write, read_array)
string = idx2numpy.convert_to_string(read_array)

# Image
# Label which is just a digit


class Image:
    pixels = np.array
    num = int


class Cell:
    def __init__(self, num):
        self.num = num
    weights = np.array
    output = np.int
# we have Layer(s) in Neural Network which contains some Cells


class Layer:
    cells = []
    for i in range(10):
        x = Cell(i)
        cells.append(x)


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

