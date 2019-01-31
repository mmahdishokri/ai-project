import numpy as np
from project import Layer
from project import Cell
from project import Image


def calc_cell_output(c, im):
    return np.dot(c.weights, im.pixels) / len(c.weights)


def get_f(im):
    f = np.array([0]*10)
    f[im.num] = 1
    return f


def mira_update_cell_weights(l, im, max_cell):
    if im.num == max_cell.num:
        return

    f = get_f(im)
    taw = (np.dot(max_cell.output - l.cells[im.num].output, f) + 1) / (2 * np.dot(f, f))

    l.cells[max_cell.num].weights -= taw * f
    l.cells[im.num].weights += taw * f


def perceptron_update_cell_weights(l, im, max_cell):
    f = get_f(im)
    l.cells[max_cell.num].weights -= f
    l.cells[im.num].weights += f


def train_cell(c, im):
    im = Image(im)
    mira_update_cell_weights(c, im)


def train_layer(l, im):
    max_cell = l.cells[0]
    for c in l.cells:
        c.output = calc_cell_output(c, im)
        if c.output > max_cell.output:
            max_cell = c
    perceptron_update_cell_weights(l, im, max_cell)
    #   mira_update_cell_weights(l, im, max_cell)
