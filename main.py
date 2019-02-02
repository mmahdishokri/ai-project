from train import train_layer, write_layer_to_json, load_layer_from_json, get_output, get_max_and_min_weights
from project import Image, get_training_data, Layer, get_testing_data
import numpy as np


def start_training_the_layer(layer, algorithm):
    train_images, train_labels = get_training_data()
    print("Training with " + algorithm + ' algorithm on '+ str(len(train_images)) + " pictures!")
    print("Starting training...")
    for i in range(len(train_images)):
        image = Image()
        image.pixels = np.ravel(train_images[i])/256
        image.num = int(train_labels[i])
        if i % 600 == 0:
            print("Training is", str(int(i/600)) + "% completed." )
        train_layer(layer, image, algorithm)


def test_trained_data(layer):
    test_images, test_labels = get_testing_data()
    error = 0
    print("Testing on " + str(len(test_images)) + " images!")
    cnt = 0
    for i in range(len(test_images)):
        image = Image()
        cnt = cnt + 1
        image.pixels = np.ravel(test_images[i])
        image.num = int(test_labels[i])
        if i % 100 == 0:
            print("Testing is", str(int(i/100)) + "% completed." )
        guess = get_output(layer, image)
        if int(guess) != int(test_labels[i]):
            error = error + 1
    print("Error = %.2f%%" % (error/cnt*100))


testing_mode = True
training_mode = True
algorithm = 'mira'
storage_file_path = 'trained-data-' + algorithm


if training_mode:
    layer = Layer()
    print("Layer created!")
    start_training_the_layer(layer, algorithm)
    print("Training is finished!!")
    print('Writing trained data to the .json file...')
    write_layer_to_json(storage_file_path, layer)
    print("data is written in json file!")
    mx, mn = get_max_and_min_weights(layer)
    print("mx = " + str(mx) + ", mn = " + str(mn))
if testing_mode:
    print('Reading trained data from the .json file...')
    layer = load_layer_from_json(storage_file_path)
    print('Data loaded successfully!')
    test_trained_data(layer)
