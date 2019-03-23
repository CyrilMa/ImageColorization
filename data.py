import os, random, skimage, scipy

import numpy as np
from skimage import io, color, transform

import keras
from keras.preprocessing.image import ImageDataGenerator

from config import dataset_conf

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, dataset = "SUN2012", training = True, batch_size=32, dim=(224,224), shuffle=True):
        'Initialization'
        self.dim = dim
        self.training = training
        self.batch_size = batch_size
        self.labels = {k:i for i, k in enumerate(dataset_conf[dataset]["labels"])}
        self.list_IDs = list_IDs
        self.n_channels = 3
        self.n_classes = dataset_conf[dataset]["n_classes"]
        self.shuffle = shuffle
        self.on_epoch_end()
        self.ABS_PATH = dataset_conf[dataset]["path"]
        self.data_gen = ImageDataGenerator(
                                rotation_range=40,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                fill_mode='nearest')

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, 1))
        y_col = np.empty((self.batch_size, *self.dim, 2), dtype=float)
        y_lab = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            im = randomCrop(io.imread(self.ABS_PATH + ID), 224, 224)
            if self.training == True:
                im = self.data_gen.flow(im.reshape((1,224,224,3)), batch_size=1).__getitem__(0)[0]/255
            else:
                im = im/255

            X[i,] = color.rgb2gray(im).reshape((224,224,1))
            
            img_lab = color.rgb2lab(im)
            img_lab = (img_lab + 128) / 255
            img_ab = img_lab[:, :, 1:3]

            y_col[i,] = img_ab
            y_lab[i] = self.labels[ID.split("/")[1]]

        return (X, [y_col,keras.utils.to_categorical(y_lab, num_classes=self.n_classes)])


# Data Augmentation Utils

def rgb2gray(rgb):
  gray = np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
  return gray.reshape(*gray.shape,1)

def randomCrop(img, width, height):
    if img.shape[0] <= height:
      img = transform.rescale(img, height/img.shape[0])
    if img.shape[1] <= width:
      img = transform.rescale(img, width/img.shape[1])    
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    return img

# Build training and validation set

def build_dataset(dataset = "SUN2012"):
    ABS_PATH = dataset_conf[dataset]["path"]
    labels = dataset_conf[dataset]["labels"]
    train_set, val_set = [], []
    if dataset == "SUN2012":
        for path in os.listdir(ABS_PATH):
            if len(path)!=1:
                continue
            for subpath in os.listdir(ABS_PATH+path):
                if subpath not in labels:
                    continue
                for im in os.listdir(ABS_PATH+path+"/"+subpath):
                    if ".jpg" not in im:
                        continue
                    if io.imread(ABS_PATH+path+"/"+subpath+"/"+im).shape[-1]!=3:
                        continue
                    if random.random() < 0.05:
                        val_set.append(path+"/"+subpath+"/"+im)
                    else:
                        train_set.append(path+"/"+subpath+"/"+im)

    if dataset == "MIT-places":
        for path in os.listdir(ABS_PATH):
            for im in os.listdir(ABS_PATH+path):
                if ".jpg" not in im:
                    continue
                if io.imread(ABS_PATH+path+"/"+im).shape[-1]!=3:
                    continue
                if random.random() < 0.05:
                    val_set.append(path+"/"+im)
                else:
                    train_set.append(path+"/"+im)

    with open(ABS_PATH+'training_set.txt', 'w') as f:
        for item in train_set:
            f.write("%s\n" % item)

    with open(ABS_PATH+'validation_set.txt', 'w') as f:
        for item in val_set:
            f.write("%s\n" % item)

def load_dataset(dataset = "SUN2012"):
    ABS_PATH = dataset_conf[dataset]["path"]

    training_set = []
    f = open(ABS_PATH+"training_set.txt",'r')
    for line in f.readlines():
        training_set.append(line[:-1])

    validation_set = []
    f = open(ABS_PATH+"validation_set.txt",'r')
    for line in f.readlines():
        validation_set.append(line[:-1])

    return training_set, validation_set


    
