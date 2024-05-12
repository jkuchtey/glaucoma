import tensorflow as tf
import matplotlib as plt
import pandas as pd
import numpy as np
import sys
import os
import shutil


def train_validate(directory, batch_size, seed, split):
    ds_training = tf.keras.preprocessing.image_dataset_from_directory( 
        directory, 
        labels = "inferred", 
        label_mode = "binary",
        class_names = ["0", "1"], 
        color_mode = 'rgb',
        image_size = (2106, 1944), 
        shuffle = True ,
        seed = seed, 
        validation_split = split, 
        subset = "both"
    )
    ds_validation = tf.keras.preprocessing.image_dataset_from_directory( 
        directory, 
        labels = "inferred", 
        label_mode = "binary",
        class_names = ["0", "1"], 
        color_mode = 'rgb',
        image_size = (2106, 1944), 
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





def main():
    directory = sys.argv[1]
    if directory == 'G1020':
        labeldata = pd.read_csv(f'G1020/G1020.csv')
        for file in os.listdir('G1020/Images'):
            shutil.copy(f'G1020/Images/{file}', f'G1020/copy/{file}')
     
        for filename in os.listdir('G1020/copy'):
            f = os.path.join('G1020/copy', filename)
            if f.endswith(".json"):
                os.remove(f)
        
        for image in os.listdir('G1020/copy'):
            imageindex =  os.listdir('G1020/copy').index(image)
            label = labeldata.loc[imageindex,"binaryLabels"]
            os.replace(f"G1020/Images/{image}", f"G1020_sorted/{label}/{image}")

        

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

    #     return networkG1020/Images
    





if __name__ == "__main__":
    main()