import numpy as np


class Cell:
    weights = np.array()


def calc_cell_output(c, input):
    input = list(input)
    return np.dot(c.weights, input) / len(c.weights)


def mira_update_cell_weights(c, target):
    #TODO
    return c


def train_cell(c, im, target):
    c = Cell(c)
    im = list(im)
    target = int(target)

    mira_update_cell_weights(c, target)


def train_layer(l, im, target):
    max_cell = l[0]
    for c in l:
        calc_cell_output(c, im)
        if c.output > max_cell:
            max_cell = c
