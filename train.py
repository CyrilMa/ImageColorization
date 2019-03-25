import os, sys

from data import DataGenerator, Equalizer, build_dataset, load_dataset
from nn import ImageColorizer2
from config import dataset_conf

import keras
import tensorflow as tf
config = tf.ConfigProto( device_count = {'GPU': 1} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)


dataset = "short_SUN2012"
if len(sys.argv) >= 2: # weights filename without path
    dataset = sys.argv[1]
ABS_PATH = dataset_conf[dataset]["path"]

weights = None
if len(sys.argv) >= 3: # weights filename without path
    weights = sys.argv[2]

# Load the data
if "training_set.txt" not in os.listdir(ABS_PATH):
    print("Building Dataset...")
    build_dataset(dataset)
print("Loading Dataset...")
train_set, val_set = load_dataset(dataset)

# Prepare the data generator
equalizer = Equalizer(train_set)
equalizer.fit(dataset_conf[dataset]["path"], train_set)

train_generator = DataGenerator(train_set, dataset=dataset, equalizer = equalizer)
validation_generator = DataGenerator(val_set, dataset=dataset, training = False, equalizer = equalizer)

# Prepare the model
print("Building Model...")
colorizer = ImageColorizer2(dataset_name = dataset)
if weights:
    colorizer.load_weights("weights/"+weights)

# Let's train
colorizer.epochs = 800
train_generator.batch_size = 16
colorizer.fit(train_generator, validation_generator)