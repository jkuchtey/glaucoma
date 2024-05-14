import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os
import pathlib
import PIL
import PIL.Image

directory = str(sys.argv[1] + "_square_sorted")
batch_size = 32
seed = 1234
split = 0.7
hct = 10
lr = 0.01


ds_training = tf.keras.preprocessing.image_dataset_from_directory( 
    directory, 
    labels = "inferred", 
    label_mode = "categorical",
    class_names = ["0", "1"], 
    color_mode = 'rgb',
    batch_size = batch_size, 
    image_size = (4, 4), 
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
    image_size = (157, 128), 
    shuffle = True,
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


def create_network(hct, oct):
    hidden_layer = tf.keras.layers.Dense(hct, activation='sigmoid') 
    output_layer = tf.keras.layers.Dense(oct)

    all_layers = [hidden_layer, output_layer]
    network = tf.keras.models.Sequential(all_layers)

    return network

def train_network(network, training_X, training_y, oct, lr):
    # create the algorithm that learns the weight of the network 
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    # create the loss function function that tells optimizer how much error it has in its predictions
    if oct == 1:
        loss_function = tf.keras.losses.MeanSquaredError()
        network.compile(optimizer=optimizer, loss=loss_function, metrics=["mse"])

    else:
        loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        network.compile(optimizer=optimizer, loss=loss_function, metrics=["accuracy"])

    # prepare the network for training

    # create a logger to save the training details to file
    csv_fname  = "test.csv"
    # if oct > 1:
    #     csv_fname = "accuracy.csv"


    csv_logger = tf.keras.callbacks.CSVLogger(csv_fname)

    # train the network for 250 epochs (setting aside 20% of the training data as validation data)
    network.fit(training_X, training_y, validation_split=0.2, epochs=250, callbacks=[csv_logger])


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

# training_X, training_y = sep_x_y(ds_training)
# testing_X, testing_y = sep_x_y(ds_validation)

train_x_batches = np.concatenate([x for x, y in ds_training], axis=0)
train_y_batches = np.concatenate([y for x, y in ds_training], axis=0)

print(train_x_batches.shape)
print(train_y_batches.shape)


# x_train = ds_training.map(lambda i: i['image'])
# y_train = ds_training.map(lambda l: l['label'])
# x_test = ds_validation.map(lambda x: x['image'])
# y_test = ds_validation.map(lambda y: y['label'])



n = create_network(hct, 2)
train_network(n, train_x_batches, train_y_batches, 2, lr)



