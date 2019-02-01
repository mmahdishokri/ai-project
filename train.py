import numpy as np


def calc_cell_output(c, im):
    # return 0.5
    # print(im.pixels)
    return np.dot(c.weights, im.pixels) / len(c.weights)
    # we should insert the kernel thing instead of DOT


def mira_update_cell_weights(l, im, max_cell):
    if im.num == max_cell.num:
        return
    else:
        learning_rate = 0.01
        f = im.pixels
        taw = (np.dot(max_cell.output - l.cells[im.num].output, f) + 1) / (2 * np.dot(f, f))
        l.cells[max_cell.num].weights -= learning_rate * taw * f
        l.cells[im.num].weights += learning_rate * taw * f


def perceptron_update_cell_weights(l, im, max_cell):
    # print(l.cells[max_cell.num].weights.__sizeof__())
    # print(type(l.cells[max_cell.num].weights))
    # print(im.pixels.__sizeof__())
    # print(type(im.pixels))
    learning_rate = 0.01
    l.cells[max_cell.num].weights -= learning_rate * im.pixels / 255
    l.cells[im.num].weights += learning_rate * im.pixels / 255
    return
    # print()
    # print(l.cells[max_cell.num].weights)
    # print(l.cells[im.num].weights)


def train_layer(layer, image):
    max_cell = layer.cells[0]
    for c in layer.cells:
        c.output = calc_cell_output(c, image)
        if c.output > max_cell.output:
            max_cell = c
    # print('done the train layer')
    perceptron_update_cell_weights(layer, image, max_cell)
    #   mira_update_cell_weights(l, im, max_cell)

    # todo kernel_update_cell_weights
