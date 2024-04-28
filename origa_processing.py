import tensorflow as tf
import matplotlib as plt
import pandas as pd
import numpy as np
import sys
import os
import pathlib
import PIL
import PIL.Image

directory = sys.argv[1]

labeldf = pd.read_csv(directory + "/OrigaList.csv")

directory = directory + "/Images"

labeldf = labeldf[["Filename", "Glaucoma"]]

l = labeldf.values.tolist()

print(l)

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory, 
    labels = l, 
    label_mode = "binary", 
    class_names = None)