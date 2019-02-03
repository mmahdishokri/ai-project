import random

import numpy as np
import json

from numpy import linalg
from numpy.ma import argmax

from project import Layer, Cell


def write_data_to_json(filepath, data):
    with open(filepath + '.json', 'w') as file:
        json.dump(data, file)


def write_layer_to_json(filepath, layer):
    layer_dict = {}
    for i in range(len(layer.cells)):
        cell = layer.cells[i]
        layer_dict[i] = {}
        layer_dict[i]['num'] = cell.num
        layer_dict[i]['weights'] = list(cell.weights)
    write_data_to_json(filepath, layer_dict)


def load_data_from_json(filepath):
    with open(filepath + '.json', 'r') as file:
        return json.load(file)


def load_layer_from_json(filepath):
    layer_dict = load_data_from_json(filepath)
    layer = Layer()
    for iid, row in layer_dict.items():
        cell = Cell(row['num'])
        cell.weights = row['weights']
        layer.cells.append(cell)
    return layer


def get_max_and_min_weights(layer):
    mx, mn = 0, 0
    for cell in layer.cells:
        mx = max(mx, max(cell.weights))
        mn = min(mn, min(cell.weights))
    return mx, mn


def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))


def get_output_kernel(alpha, MAX, images, image):
    vals = [0] * 10
    for i in range(MAX):
        vals[images[i].num] += alpha[i] * gaussian_kernel(images[i].pixels, image.pixels)
    #print(image.pixels, '---->', end='')
    print(vals, end=' ')
    print(' ---> ', argmax(vals))
    return argmax(vals)


def train_layer_kernel(MAX, images):
    MAX = min(MAX, len(images))
    random.shuffle(images)
    K = np.zeros((MAX, MAX))
    alpha = [0] * MAX
    for i in range(MAX):
        for j in range(MAX):
            K[i][j] = gaussian_kernel(images[i].pixels, images[j].pixels)
        if i % int(MAX/100) == 0:
            print('first phase of training is ' + str(int(i/int(MAX/100))) + '% completed.')
    cnt = 0
    for i in range(MAX):
        image = images[i]
        ret = get_output_kernel(alpha, MAX, images, image)
        if ret != image.num:
            cnt = cnt + 1
            for j in range(MAX):
                if images[j].num == ret:
                    alpha[j] = alpha[j] - 1
                if images[j].num == image.num:
                    alpha[j] = alpha[j] + 1

        if i % int(MAX/100) == 0:
            print('second phase of training is ' + str(int(i/int(MAX/100))) + '% completed.')

    print('Learning is completed! MAX = ' + str(MAX) + 'alpha =', alpha)
    print('cnt =', cnt)
    return alpha


def calc_cell_output(c, im):
    # return 0.5
    # print(im.pixels)
    return np.dot(c.weights, im.pixels) / len(c.weights)
    # we should insert the kernel thing instead of DOT


def mira_update_cell_weights(l, im, max_cell):
    if im.num == max_cell.num:
        return
    else:
        learning_rate = 2
        f = im.pixels
        taw = (np.dot((max_cell.output - l.cells[im.num].output)*(len(im.pixels)), f) + 1) / (2 * np.dot(f, f))
        l.cells[max_cell.num].weights -= learning_rate * taw * f
        l.cells[im.num].weights += learning_rate * taw * f
        return


def perceptron_update_cell_weights(l, im, max_cell):
    # print(l.cells[max_cell.num].weights.__sizeof__())
    # print(type(l.cells[max_cell.num].weights))
    # print(im.pixels.__sizeof__())
    # print(type(im.pixels))
    learning_rate = 1
    l.cells[max_cell.num].weights -= learning_rate * im.pixels
    l.cells[im.num].weights += learning_rate * im.pixels
    return
    # print()
    # print(l.cells[max_cell.num].weights)
    # print(l.cells[im.num].weights)



def kernel_update_cell_weights(l, im, max_cell):
    # todo
    return

def get_output(layer, image):
    max_cell_num = 0
    max_cell_output = calc_cell_output(layer.cells[0], image)
    for c in layer.cells:
        cur = calc_cell_output(c, image)
        if cur > max_cell_output:
            max_cell_num = c.num
            max_cell_output = cur
    return max_cell_num


def train_layer(layer, image, algorithm):
    max_cell = layer.cells[0]
    for c in layer.cells:
        c.output = calc_cell_output(c, image)
        if c.output > max_cell.output:
            max_cell = c
    # print('done the train layer')
    if algorithm == 'perceptron':
        perceptron_update_cell_weights(layer, image, max_cell)
    if algorithm == 'mira':
        mira_update_cell_weights(layer, image, max_cell)
    if algorithm == 'kernel':
        kernel_update_cell_weights(layer, image, max_cell)
