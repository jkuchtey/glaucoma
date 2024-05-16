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

itype = "cropped"

g1020_dir = "G1020_sorted"

directory = "ORIGA_" + itype + "_sorted"
batch_size = 32
seed = 1234
split = 0.7
hct = 10
lr = 0.0001


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

def plot_bars(df, x, y, fill, title, filename):

    acc_bar = (
        ggplot(df)
        + aes(x=x, y=y, fill=fill) 
        + geom_col()
        + ggtitle(title)
    )
    acc_bar.save(filename=filename)


def plot_epochs(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    plt.savefig(fname="optimalmodelepochs.png")


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



# Create CSV to compare Batch sizes

# accs = compare_batch_size()
# print(accs)
# bs_comp_df = pd.DataFrame.from_dict(accs, orient='index')
# bs_comp_df.to_csv("batch_size_comparison_ORIGA_" + itype + ".csv")


#Plot accuracies comparing batch size

# bs_comp_df = pd.read_csv("batch_size_comparison_ORIGA_" + itype + ".csv")
# print(bs_comp_df)
# bs_comp_df["Batch Size"] = bs_comp_df["Batch Size"].astype(str)
# plot_bars(bs_comp_df, "Batch Size", "Accuracy", "Batch Size", "Batch Size Accuracy Comparison " + itype,  "bs_comparison_" + itype + ".png" )




# Create CSV of compared learning rates

# csv_filename = "lr_accuracies_ORIGA_" + itype + ".csv"

# data = tf.keras.utils.image_dataset_from_directory(
#     directory, 
#     shuffle=True, 
#     seed=seed, 
#     validation_split=split, 
#     subset="both", 
#     labels="inferred", 
#     image_size=(32, 32), 
#     label_mode="binary", 
#     class_names=["0", "1"], 
#     batch_size=32
# )

# n =create_cnn()
# accs = compare_lrs(n, data)
# accs_df = pd.DataFrame.from_dict(accs, orient='index')
# accs_df.to_csv(csv_filename)

# Plot the learning rate CSV

# lr_comp_df = pd.read_csv(csv_filename)
# lr_comp_df["Learning Rate"] = lr_comp_df["Learning Rate"].astype(str)
# plot_bars(lr_comp_df, "Learning Rate", "Accuracy", "Learning Rate", "Learning Rate Accuracy Comparison " + itype,  "lr_comparison_" + itype + ".png")


# Measure accuracy on G1020 dataset


# data = tf.keras.utils.image_dataset_from_directory(
#     directory, 
#     shuffle=True, 
#     seed=seed, 
#     validation_split=split, 
#     subset="both", 
#     labels="inferred", 
#     image_size=(32, 32), 
#     label_mode="binary", 
#     class_names=["0", "1"], 
#     batch_size=32
# )
# g1020 = tf.keras.utils.image_dataset_from_directory(
#     g1020_dir, 
#     shuffle=True, 
#     labels="inferred", 
#     image_size=(32, 32), 
#     label_mode="binary", 
#     class_names=["0", "1"], 
#     batch_size=32
# )

# n = create_cnn()
# history = train_cnn(n, data, 100, 0.001)
# test_loss, test_acc = n.evaluate(g1020, verbose=2)
# print(test_acc)


# Single Run

# data = tf.keras.utils.image_dataset_from_directory(
#     directory, 
#     shuffle=True, 
#     seed=seed, 
#     validation_split=split, 
#     subset="both", 
#     labels="inferred", 
#     image_size=(32, 32), 
#     label_mode="binary", 
#     class_names=["0", "1"], 
#     batch_size=32
# )
# history = train_cnn(n, data, 250, 0.001)
# test_loss, test_acc = n.evaluate(data[1], verbose=2)
# print(test_acc)

# plot_epochs(history)


# Single Run With G1020

# data = tf.keras.utils.image_dataset_from_directory(
#     g1020_dir, 
#     shuffle=True, 
#     seed=seed, 
#     validation_split=split, 
#     subset="both", 
#     labels="inferred", 
#     image_size=(32, 32), 
#     label_mode="binary", 
#     class_names=["0", "1"], 
#     batch_size=32
# )

# history = train_cnn(n, data, 250, 0.001)
# test_loss, test_acc = n.evaluate(data[1], verbose=2)
# print(test_acc)




