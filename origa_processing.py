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

def plot_epochs(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    plt.show()

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

    csv_logger = tf.keras.callbacks.CSVLogger(csv_fname)

    # train the network for 250 epochs (setting aside 20% of the training data as validation data)
    network.fit(training_X, training_y, validation_split=0.2, epochs=250, callbacks=[csv_logger])



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

#CNN
# cnn_network = create_cnn(training_X, training_y)
# train_network(cnn_network, training_X, training_y, 20)
