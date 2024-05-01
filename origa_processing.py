import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os
import pathlib
import PIL
import PIL.Image

directory = str(sys.argv[1] + "_sorted")
batch_size = 32
seed = 1234
split = 0.7


ds_training = tf.keras.preprocessing.image_dataset_from_directory( 
    directory, 
    labels = "inferred", 
    label_mode = "binary",
    class_names = ["0", "1"], 
    color_mode = 'rgb',
    batch_size = batch_size, 
    image_size = (2512, 2048), 
    shuffle = True ,
    seed = seed, 
    validation_split = split, 
    subset = "training"
)

ds_validation = tf.keras.preprocessing.image_dataset_from_directory( 
    directory, 
    labels = "inferred", 
    label_mode = "binary",
    class_names = ["0", "1"], 
    color_mode = 'rgb',
    batch_size = batch_size, 
    image_size = (2512, 2048), 
    shuffle = True ,
    seed = seed, 
    validation_split = (1 - split), 
    subset = "validation"
)

# displays a given image from the training set along with its label
def view_image(index, training_X, training_y, label_names):
    # get the label and image from the training set
    label_num = training_y[index]
    if len(label_num.shape) > 0:
        label = label_names[training_y[index][0]]
    else:
        label = label_names[training_y[index]]
    image = training_X[index]

    # show the label then the image
    print("Label:", label)
    plt.imshow(image)
    plt.show()


def sep_x_y(ds):
    set_X = []
    set_y = []

    for x, y in ds:
        if not set_X:
            set_X = [x]
        else:
            set_X = set_X.append(x)
        if not set_y:
            set_y = [y]
        else:
            set_y = set_y.append(y)

    return set_X, set_y


training_X, training_y = sep_x_y(ds_training)
testing_X, testing_y = sep_x_y(ds_validation)


class_names = ds_training.class_names
print(class_names)

for i in range(0, len(class_names)-1):
    view_image(i, training_X, training_y, class_names)

