import os, sys

from data import DataGenerator, build_dataset, load_dataset
from nn import ImageColorizer
from config import dataset_conf

dataset = "SUN2012"
ABS_PATH = dataset_conf[dataset]["path"]

weights = None
if len(sys.argv) >= 2: # weights filename without path
    weights = sys.argv[1]

# Load the data
if "training_set.txt" not in os.listdir(ABS_PATH):
    build_dataset(dataset)
train_set, val_set = load_dataset(dataset)

# Prepare the data generator
train_generator = DataGenerator(train_set)
validation_generator = DataGenerator(val_set)

# Prepare the model
colorizer = ImageColorizer(dataset_name = dataset)
if weights:
    colorizer.load_weights("weights/"+weights)

# Let's train
colorizer.epochs = 300
colorizer.fit(train_generator, validation_generator)