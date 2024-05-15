import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os
import pathlib
import PIL
import PIL.Image
import pprint
from plotnine import *

directory = "ORIGA_square_sorted"
batch_size = 32
seed = 1234
split = 0.25
hct = 10
lr = 0.01

# Scale images

def scale(data):

    # testing_X = np.empty([batch_ct * batch_size, 4])
    # testing_y = np.empty([batch_ct * batch_size, 1])

    # iterator = next(iter(data))
    # batch_ct = len(data)



    # tx = np.empty([batch_ct * batch_size, 4])
    # ty = np.empty([batch_ct * batch_size, 1])
    # for i in range(batch_ct):
    #     for x, y in iterator:
    #         # print(x.numpy().shape, "\n\n")
    #         np.append(tx, x.numpy())
    #         #training_X.np.append(x.numpy())
    #         np.append(ty, y.numpy())
    #         #training_y.append(y.numpy())

    #     # for x, y in iterator:
    #     #     np.append(testing_X, x.numpy())
    #     #     # testing_X.append(x.numpy())
    #     #     np.append(testing_y, y.numpy())
    #     #     # testing_y.append(y.numpy())
    # iterator = next(iter(data))

    # training_X = data[0].map(lambda x,y: (x/255, y))
    # training_y = data[0].map(lambda x, y: y)
    
    # testing_X = data[1].map(lambda x,y: (x/255, y))
    # testing_y = data[1].map(lambda x, y: y) 

    for x, y in data[0]:
        training_X = x/255
        training_y = y
    for x, y in data[1]:
        testing_X = x/255
        testing_y = y

    
    return training_X, training_y, testing_X, testing_y



# for i in [0, 1]:
#     ds = data[i].map(lambda xn =create_cnn()
# history = train_cnn(n, data, 100)g

def show_imgs(data):
    data_iterator = data.as_numpy_iterator()
    batch = data_iterator.next()
    fig, ax = plt.subplots(ncols=5, figsize=(20,20))


    #show images
    for idx, img in enumerate(batch[0][:5]):
        ax[idx].imshow(img)
        ax[idx].title.set_text(batch[1][idx])
    batch = data_iterator.next()

    plt.show()
#Seperate labels and attributes

# print(data[0].class_names)


def create_model(num_classes):
    model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
    ])

    model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

    return model

def train_model(model, training, testing, epochs):
    csv_fname  = "epoch_log.csv"

    csv_logger = tf.keras.callbacks.CSVLogger(csv_fname)

    model.fit(
        training,
        validation_data=testing,
        epochs=epochs, 
        callbacks=[csv_logger]
    )

def lab8_cnn(training_X, training_y):
    # get the shape of each image so the the first layer knows what inputs it will receive
    image_shape = training_X.shape[1:]

    # if the image was grayscale, add a 1 to the end of the shape to make it 3D
    if len(image_shape) == 2:
        image_shape = (image_shape[0], image_shape[1], 1)

    # get the number of possible labels (since this is a classification task)
    num_labels = len(np.unique(testing_y))
    
    # create the layers
    conv1 = tf.keras.layers.Conv2D(16, (3, 3), activation = "relu", input_shape=image_shape)
    pool1 = tf.keras.layers.MaxPooling2D((2, 2))
    flat = tf.keras.layers.Flatten()
    dense = tf.keras.layers.Dense(128)
    out = tf.keras.layers.Dense(num_labels)

    # convert the layers into a neural network model
    layers = [conv1, pool1, flat, dense, out]    
    # layers = [conv1, pool1, conv2, flat, dense, out]    
    # layers = [conv1, pool1, conv2, pool2, conv3, flat, dense, out]    
    network = tf.keras.models.Sequential(layers)

    return network

def train_network(network, training_X, training_y, epochs):
    # create the algorithm that learns the weight of the network (with a learning rate of 0.0001)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    
    # create the loss function function that tells optimizer how much error it has in its predictions
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    # prepare the network for training
    network.compile(optimizer=optimizer, loss=loss_function, metrics=["accuracy"])
    
    # create a logger to save the training details to file
    csv_logger = tf.keras.callbacks.CSVLogger('epochs.csv')
    
    # train the network for 20 epochs (setting aside 20% of the training data as validation data)
    network.fit(training_X, training_y, validation_split=0.2, epochs=epochs, callbacks=[csv_logger])


def create_cnn():
    model = models.Sequential()
    tf.keras.layers.Rescaling(1./255)
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))


    return model

def train_cnn(model, data, epochs, lr):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    csv_logger = tf.keras.callbacks.CSVLogger('epochs.csv')

    history = model.fit(
        data[0],
        epochs=epochs, 
        validation_data = data[1], 
        callbacks = [csv_logger]
    )

    return history

def compare_lrs(n, data):
    accs = {"Learning Rate": "Accuracy"}
    for lr in [0.0001, 0.001, 0.01, 0.1]:
        history = train_cnn(n, data, 100, lr)
        test_loss, test_acc = n.evaluate(data[1], verbose=2)

        accs[lr] = test_acc

    return accs

def plot_lrs(df):
    print(df)
    acc_bar = (
        ggplot(df)
        + aes(x="Learning Rate", y="Accuracy", color="factor(Learning Rate)") 
        + geom_col()
        + ggtitle("Learning Rate Comparison")
    )
    acc_bar.save(filename="lr_comparison.png")


def compare_batch_size():
    accs = {"Batch Size": "Accuracy"}
    for bs in [10, 20, 32, 50, 70]:
        data = tf.keras.utils.image_dataset_from_directory(
            directory, 
            shuffle=True, 
            seed=seed, 
            validation_split=split, 
            subset="both", 
            labels="inferred", 
            image_size=(32, 32), 
            label_mode="binary", 
            class_names=["0", "1"], 
            batch_size=bs
        )
        history = train_cnn(n, data, 100, 0.001)
        test_loss, test_acc = n.evaluate(data[1], verbose=2)
        accs[bs] = test_acc

    return accs

# accs = compare_batch_size()
# print(accs)

data = tf.keras.utils.image_dataset_from_directory(
    directory, 
    shuffle=True, 
    seed=seed, 
    validation_split=split, 
    subset="both", 
    labels="inferred", 
    image_size=(32, 32), 
    label_mode="binary", 
    class_names=["0", "1"], 
    batch_size=32
)
# history = train_cnn(n, data, 100, 0.001)
# test_loss, test_acc = n.evaluate(data[1], verbose=2)
# print(test_acc)

# n =create_cnn()
# accs = compare_lrs(n, data)
# accs_df = pd.DataFrame.from_dict(accs, orient='index')
# accs_df.to_csv("origa_accuracies.csv")


plot_lrs(pd.read_csv("origa_accuracies.csv"))



def plot_epochs(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    plt.show()




# training_X, training_y, testing_X, testing_y = scale(data)
# print(len(training_X))






# Sequential Model
# model = create_model(2)
# train_model(model, data[0], data[1], 3)

#CNN
# cnn_network = create_cnn(training_X, training_y)
# train_network(cnn_network, training_X, training_y, 20)

