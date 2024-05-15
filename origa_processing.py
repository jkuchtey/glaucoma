import tensorflow as tf
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

directory = "ORIGA_cropped_sorted"
batch_size = 32
seed = 1234
split = 0.7
hct = 10
lr = 0.01

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

def create_cnn():
    model = models.Sequential()
    tf.keras.layers.Rescaling(1./255)
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MoraxPooling2D((2, 2)))
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

def compare_lrs(n, data, lr):
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
        + aes(x="Learning Rate", y="Accuracy", fill="Learning Rate")
        + geom_col()
        + ggtitle("Learning Rate Comparison")
        + scale_alpha()
    )
    acc_bar.save(filename="lr_comparison.png")

n =create_cnn()
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
history = train_cnn(n, data, 100, 0.001)
test_loss, test_acc = n.evaluate(data[1], verbose=2)
print(test_acc)
# accs = compare_lrs()
# accs_df = pd.DataFrame.from_dict(accs, orient='index')
# plot_lrs(accs_df)



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

