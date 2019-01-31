import numpy as np
from project import Layer
from project import Cell
from project import Image

def calc_cell_output(c, im):
    im = Image(im)
    c = Cell(c)
    return np.dot(c.weights, input) / len(c.weights)


def get_f(im):
    im = Image(im)
    f = np.array([0]*10)
    f[im.num] = 1
    return f


def mira_update_cell_weights(l, im, max_cell):
    if im.num == max_cell.num:
        return

    f = get_f(im)
    taw = (np.dot(max_cell.output - l[im.num].output, f) + 1) / (2 * np.dot(f, f))

    l[max_cell.num].weights -= taw * f
    l[im.num].weights += taw * f


def perceptron_update_cell_weights(l, im, max_cell):
    f = get_f(im)
    l[max_cell.num].weights -= f
    l[im.num].weights += f


def train_cell(c, im):
    im = Image(im)
    mira_update_cell_weights(c, im)


def train_layer(l, im):
    l = Layer(l)
    max_cell = l[0]
    for c in l:
        calc_cell_output(c, im)
        if c.output > max_cell:
            max_cell = c
    mira_update_cell_weights(l, im, max_cell)
    #   perceptron_update_cell_weights(l, im, max_cell)
