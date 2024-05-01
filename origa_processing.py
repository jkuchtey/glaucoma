import tensorflow as tf
import matplotlib as plt
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
    csv_fname  = "est.csv"
    # if oct > 1:
    #     csv_fname = "accuracy.csv"


    csv_logger = tf.keras.callbacks.CSVLogger(csv_fname)

    # train the network for 250 epochs (setting aside 20% of the training data as validation data)
    network.fit(training_X, training_y, validation_split=0.2, epochs=250, callbacks=[csv_logger])

#n= create_network(10, 2)

print(ds_training)