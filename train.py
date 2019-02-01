import numpy as np
from project import Layer
from project import Cell
from project import Image


def calc_cell_output(c, im):
    return np.dot(c.weights, im.pixels) / len(c.weights)
    # we should insert the kernel thing instead of DOT


def mira_update_cell_weights(l, im, max_cell):
    if im.num == max_cell.num:
        return

    f = im.pixels
    taw = (np.dot(max_cell.output - l.cells[im.num].output, f) + 1) / (2 * np.dot(f, f))

    l.cells[max_cell.num].weights -= taw * f
    l.cells[im.num].weights += taw * f
    # TODO: learning rate


def perceptron_update_cell_weights(l, im, max_cell):
    l.cells[max_cell.num].weights -= im.pixels
    l.cells[im.num].weights += im.pixels
    # TODO: learning rate, time vary


def train_layer(layer, image):
    max_cell = layer.cells[0]
    for c in layer.cells:
        c.output = calc_cell_output(c, image)
        if c.output > max_cell.output:
            max_cell = c

    perceptron_update_cell_weights(layer, image, max_cell)
    #   mira_update_cell_weights(l, im, max_cell)

    # todo kernel_update_cell_weights
