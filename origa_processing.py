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


# ds = tf.keras.preprocessing.image_dataset_from_directory( 
#     directory, 
#     labels = "inferred", 
#     label_mode = "binary",
#     class_names = ["0", "1"], 
#     color_mode = 'rgb',
#     batch_size = batch_size, 
#     image_size = (10, 10), 
#     shuffle = True ,
#     seed = seed, 
#     validation_split = split, 
#     subset = "both", 
# )

data = tf.keras.utils.image_dataset_from_directory(
    directory, 
    shuffle=True, 
    seed=seed, 
    validation_split=split, 
    subset="both", 
    labels="inferred", 
    label_mode="binary", 
    class_names=["0", "1"], 
    batch_size=batch_size)





#Scale images
tt = []
for i in [0, 1]:
    ds = data[i].map(lambda x,y: (x/255, y))
    tt.append(ds)
    data[i].as_numpy_iterator().next()

training = tt[0]
testing = tt[1]

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

print(data[0].class_names)


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
    model.fit(
    training,
    validation_data=testing,
    epochs=epochs
)

model = create_model(2)
train_model(model, data[0], data[1], 50)

# training_X, training_y = sep_x_y(ds_training)
# testing_X, testing_y = sep_x_y(ds_validation)

# train_x_batches = np.concatenate([x for x, y in ds_training], axis=0)
# train_y_batches = np.concatenate([y for x, y in ds_training], axis=0)

# y_test = ds_validation.map(lambda y: y['label'])



# n = create_network(hct, 2)
# train_network(n, train_x_batches, train_y_batches, 2, lr)



training