import idx2numpy
import numpy as np

# read_array = idx2numpy.convert_from_file('myfile.idx')
# f_read = open('myfile.idx', 'rb')
#
# read_array = idx2numpy.convert_from_file(f_read)
# string = f_read.read()
# read_array = idx2numpy.convert_from_string(string)
#
#
# idx2numpy.convert_to_file('myfile_copy.idx', read_array)
# f_write = open('myfile_copy2.idx', 'wb')
# idx2numpy.convert_to_file(f_write, read_array)
# string = idx2numpy.convert_to_string(read_array)
#
# Image
# Label which is just a digit


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
        x = Cell(i)
        cells.append(x)


def announce_output(layer):
    #   todo print the max
    return Cell(1)


def announce_error():
    #   todo the sum of the Squares
    return Cell(1)
