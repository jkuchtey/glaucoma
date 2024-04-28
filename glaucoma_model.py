import tensorflow as tf
import matplotlib as plt
import pandas as pd
import numpy as np
import sys
import os

directory = sys.argv[1]


if directory == 'G1020':

    labeldata = pd.read_csv('G1020.csv')
    for label in labeldata:
        for image in os.listdir('GLAUCOMA/G1020/Images'):
            if image.endswith(".jpg"):


                tf.keras.preprocessing.image_dataset_from_directory(
                    directory,
                    labels='inferred',
                    label_mode="binary",
                    class_names=None,
                    color_mode='rgb',
                    batch_size=32,
                    image_size=(2656, 2048),
                    shuffle=True,
                    seed=None,
                    validation_split=None,
                    subset=None,
                    interpolation='bilinear',
                    follow_links=False,
                    crop_to_aspect_ratio=False,
                    pad_to_aspect_ratio=False,
                    data_format=None,
                    verbose=True
                )

# def create_cnn(training_X, training_y):
#     # get the shape of each image so the the first layer knows what inputs it will receive
#     image_shape = training_X.shape[1:]

#     # if the image was grayscale, add a 1 to the end of the shape to make it 3D
#     if len(image_shape) == 2:
#         image_shape = (image_shape[0], image_shape[1], 1)

#     # get the number of possible labels (since this is a classification task)
#     num_labels = len(np.unique(testing_y))
    
#     # create the layers
#     conv1 = tf.keras.layers.Conv2D(16, (3, 3), activation = "relu", input_shape=image_shape)
#     pool1 = tf.keras.layers.MaxPooling2D((2, 2))
#     flat = tf.keras.layers.Flatten()
#     dense = tf.keras.layers.Dense(128)
#     out = tf.keras.layers.Dense(num_labels)

#     # convert the layers into a neural network model
#     layers = [conv1, pool1, flat, dense, out]    
#     # layers = [conv1, pool1, conv2, flat, dense, out]    
#     # layers = [conv1, pool1, conv2, pool2, conv3, flat, dense, out]    
#     network = tf.keras.models.Sequential(layers)

#     return network