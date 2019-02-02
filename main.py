from train import train_layer, write_layre_to_json
from project import Image, get_training_data, Layer
import numpy as np


def start_training_the_layer(layer):
    train_images, train_labels = get_training_data()
    print("Length:", len(train_images))
    print("Starting training...")
    for i in range(len(train_images)):
        image = Image()
        image.pixels = np.ravel(train_images[i])
        image.num = int(train_labels[i])
        if i % 600 == 0:
            print("Training is ", int(i/600), "% completed." )
        train_layer(layer, image)


layer = Layer()
print("Layer created!")
start_training_the_layer(layer)
print("Training is finished!!")

write_layre_to_json('trained-data.json', layer)

print("data is written in json file!")