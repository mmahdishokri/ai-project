from train import train_layer
from project import Image, get_training_data, Layer
import numpy as np


def start_training_the_layer(layer):
    train_images, train_labels = get_training_data()
    for i in range(train_images.size):
        image = Image()
        # im = np.concatenate(train_images[i], axis=0)
        # pix = image.pixels
        # print('pix: ', type(pix))
        # print(pix)
        image.pixels = np.ravel(train_images[i])
        image.num = train_labels[i]
        train_layer(layer, image)


layer = Layer()
start_training_the_layer(layer)
print(layer.cells[0].weights)
