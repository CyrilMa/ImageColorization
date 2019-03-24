import numpy as np

import tensorflow as tf
import keras
from keras.layers import Input, Dense, UpSampling2D, Reshape, Flatten, Dropout, Activation, ZeroPadding2D, MaxPooling2D,Concatenate, Lambda
from keras.layers import BatchNormalization as BN
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.backend import tile, shape

from config import dataset_conf

class ImageColorizer():

    def __init__(self,dataset_name='SUN2012', image_size = (224,224)):

        #X_train = self.load_data(dataset_name)
        optimizer = 'adadelta'#Adam(0.0002, 0.5) #

        # image parameters
        self.epochs = 200
        self.error_list = np.zeros((self.epochs,1))
        self.img_rows = image_size[0]
        self.img_cols = image_size[1]
        self.img_channels = 3
        self.img_shape = (*image_size, 1)
        self.dataset_name = dataset_name
        self.n_classes = dataset_conf[dataset_name]["n_classes"]

        # Build and compile the autoencoder
        self.nn = self.build_nn()
        self.nn.summary()
        
        self.nn_style_transfer = self.build_nn_style_transfer()
        self.nn_style_transfer.summary()
        #binary cross-entropy loss, because mnist is grey-scale
        #you can try out the mse loss as well if you like
        self.nn.compile(optimizer=optimizer, 
                        loss={'fuse_and_colorize_model': 'mse', 'classification_output': 'categorical_crossentropy'}, 
                        loss_weights = {'fuse_and_colorize_model': 1, 'classification_output':1/300}
                       )

    def build_nn(self):
        # For train network we take a fixed size
        
        input_img = Input(shape=self.img_shape, name = "Input")
        self.build_convolutional_feature_extractor()
        self.build_global_features_network()
        self.build_fuse_and_colorize_network()
        self.build_mid_level_feature_extractor()
        
        # Global Features Network
        dense_gf_3, dense_gf_2, conv_ll_6 = self.cnn_global_features(input_img)
        
        # Classification Network
        
        dense_cl_1 =  Dropout(0.5)(Dense(256, activation="relu", name="dense_cl_1")(dense_gf_2))
        classification_output = Dense(self.n_classes, activation="softmax", name="classification_output")(dense_cl_1)

        # Mid Level Features Network
        conv_ml_2 = self.cnn_mid_features(conv_ll_6)
        
        # Fuse and colorize
        colorization_output = self.cnn_fuse_and_colorize([dense_gf_3, conv_ml_2])
       

        return Model(input_img, [colorization_output, classification_output])
    
    def build_nn_style_transfer(self):
        # for test network we allow arbitray sizes
        
        img_to_colorize = Input(shape=(None, None, 1), name = "img_to_colorize")
        img_with_style = Input(shape=self.img_shape, name = "img_with_style")
        
        # Global Features Network
        dense_gf_3, _,_ = self.cnn_global_features(img_with_style)

        # Mid Level Features Network
        conv_ll_6_img_to_colorize = self.cnn_feature_extractor(img_to_colorize)
        conv_ml_2 = self.cnn_mid_features(conv_ll_6_img_to_colorize)
        
        # Fuse and colorize
        colorization_output = self.cnn_fuse_and_colorize([dense_gf_3, conv_ml_2])
       

        return Model([img_to_colorize, img_with_style], colorization_output, name="style_transfer_model")
    
    def build_convolutional_feature_extractor(self):
        input_img = Input(shape=(None,None,1), name = "feature_extractor")
        
        conv_ll_1 = BN()(Conv2D(64, 3, strides = (2,2), padding="same", activation="relu", name="conv_ll_1")(input_img))
        conv_ll_2 = BN()(Conv2D(128, 3, strides = (1,1), padding="same", activation="relu", name="conv_ll_2")(conv_ll_1))
        
        conv_ll_3 = BN()(Conv2D(128, 3, strides = (2,2), padding="same", activation="relu", name="conv_ll_3")(conv_ll_2))
        conv_ll_4 = BN()(Conv2D(256, 3, strides = (1,1), padding="same", activation="relu", name="conv_ll_4")(conv_ll_3))
        
        conv_ll_5 = BN()(Conv2D(256, 3, strides = (2,2), padding="same", activation="relu", name="conv_ll_5")(conv_ll_4))
        conv_ll_6 = BN()(Conv2D(512, 3, strides = (1,1), padding="same", activation="relu", name="conv_ll_6")(conv_ll_5))
        
        self.cnn_feature_extractor = Model(input_img, conv_ll_6, name="shared_feature_extractor")
        
    def build_global_features_network(self):
        input_img = Input(shape=self.img_shape, name = "Input_global")
        
#         global_input_img = Reshape(target_shape=self.img_shape)(input_img)
        global_conv_ll_6 = self.cnn_feature_extractor(input_img)
        
        conv_gf_1 = BN()(Conv2D(512, 3, strides = (2,2), padding="same", activation="relu", name="conv_gf_1")(global_conv_ll_6))
        conv_gf_2 = BN()(Conv2D(512, 3, strides = (1,1), padding="same", activation="relu", name="conv_gf_2")(conv_gf_1))
        
        conv_gf_3 = BN()(Conv2D(512, 3, strides = (2,2), padding="same", activation="relu", name="conv_gf_3")(conv_gf_2))
        conv_gf_4 = BN()(Conv2D(512, 3, strides = (1,1), padding="same", activation="relu", name="conv_gf_4")(conv_gf_3))
        
        flatten_gf = BN()(Flatten()(conv_gf_4))
        
        dense_gf_1 = BN()(Dense(1024, name="dense_gf_1")(flatten_gf))
        dense_gf_2 = BN()(Dense(512, name="dense_gf_2")(dense_gf_1))
        dense_gf_3 = BN()(Dense(256, name="dense_gf_3")(dense_gf_2))
        
        self.cnn_global_features = Model(input_img, [dense_gf_3, dense_gf_2, global_conv_ll_6], name="global_features_extractor")
        
    def build_fuse_and_colorize_network(self):
        dense_gf_3 = Input(shape=(256,), name = "global_features")
        conv_ml_2 = Input(shape=(None, None, 256), name = "mid_level_features")
        
         # Fusion Network
        repeated_global_features = Lambda(self.repeat_vector, output_shape=(None, None, 256)) ([dense_gf_3, conv_ml_2])
        concat_fn = Concatenate(axis=-1, name="concat_fn")([repeated_global_features, conv_ml_2])
        
        # Colorization Network
        
        conv_cl_1 = BN()(Conv2D(128, 3, strides = (1,1), padding="same", activation="relu", name="conv_cl_1")(concat_fn))
        upsample_1 = UpSampling2D(name="upsample_1")(conv_cl_1)
                          
        conv_cl_2 = BN()(Conv2D(64, 3, strides = (1,1), padding="same", activation="relu", name="conv_cl_2")(upsample_1))
        conv_cl_3 = BN()(Conv2D(64, 3, strides = (1,1), padding="same", activation="relu", name="conv_cl_3")(conv_cl_2))
        upsample_2 = UpSampling2D(name="upsample_2")(conv_cl_3) 
                          
        conv_cl_4 = BN()(Conv2D(32, 3, strides = (1,1), padding="same", activation="relu", name="conv_cl_4")(upsample_2))
        conv_cl_5 = BN()(Conv2D(2, 3, strides = (1,1), activation="sigmoid", padding="same", name="output_cl")(conv_cl_4))
        colorization_output = UpSampling2D(name="colorization_output")(conv_cl_5)
        
        self.cnn_fuse_and_colorize = Model([dense_gf_3, conv_ml_2], colorization_output, name="fuse_and_colorize_model")
        
    def build_mid_level_feature_extractor(self):
        mid_conv_ll_6 = Input(shape=(None, None, 512), name = "shared_features_extracted")
        conv_ml_1 = BN()(Conv2D(512, 3, strides = (1,1), padding="same", activation="relu", name="conv_ml_1")(mid_conv_ll_6))
        conv_ml_2 = BN()(Conv2D(256, 3, strides = (1,1), padding="same", activation="relu", name="conv_ml_2")(conv_ml_1))
        
        self.cnn_mid_features = Model(mid_conv_ll_6, conv_ml_2, name="mid_features_extractor")
        
    def repeat_vector(self, args):
        # Inspiré de https://github.com/keras-team/keras/issues/7949
        
        layer_to_repeat = args[0]
        sequence_layer = args[1]
        res = tile(Reshape((1,1,256))(layer_to_repeat), (1,shape(sequence_layer)[1],shape(sequence_layer)[2],1))
        return res
    
    # def resize_img(self, input_tensor):
    #     input_tensor = args
    
    def load_weights(self, path):
        self.nn.load_weights(path)

    def fit(self, train, val): # train, val DataGenerator

        filepath="weights/weight-%s-{epoch:02d}-{val_loss:.4f}.hdf5"%self.dataset_name
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        
        self.nn.fit_generator(train, epochs = self.epochs, validation_data=val, callbacks = [checkpoint])        
    
    def predict(self, test):
        self.nn.predict(test)
        
    def transfer_style(self, img_to_colorize, img_with_style):
        self.nn_style_transfer([img_to_colorize, img_with_style])


class ImageColorizer2():

    def __init__(self,dataset_name='SUN2012', image_size = (224,224)):

        #X_train = self.load_data(dataset_name)
        optimizer = 'adadelta'#Adam(0.0002, 0.5) #

        # image parameters
        self.epochs = 200
        self.error_list = np.zeros((self.epochs,1))
        self.img_rows = image_size[0]
        self.img_cols = image_size[1]
        self.img_channels = 3
        self.img_shape = (*image_size, 1)
        self.dataset_name = dataset_name
        self.n_classes = dataset_conf[dataset_name]["n_classes"]

        # Build and compile the autoencoder
        self.nn = self.build_nn()
        self.nn.summary()
        
        self.nn_style_transfer = self.build_nn_style_transfer()
        self.nn_style_transfer.summary()
        #binary cross-entropy loss, because mnist is grey-scale
        #you can try out the mse loss as well if you like
        self.nn.compile(optimizer=optimizer, 
                        loss={'fuse_and_colorize_model': 'mse', 'classification_output': 'categorical_crossentropy'}, 
                        loss_weights = {'fuse_and_colorize_model': 1, 'classification_output':1/300}
                       )

    def build_nn(self):
        # For train network we take a fixed size
        
        input_img = Input(shape=self.img_shape, name = "Input")
        self.build_convolutional_feature_extractor()
        self.build_global_features_network()
        self.build_fuse_and_colorize_network()
        self.build_mid_level_feature_extractor()
        
        # Global Features Network
        conv_ll_2, conv_ll_4, conv_ll_6 = self.cnn_feature_extractor(input_img)
        dense_gf_3, dense_gf_2, conv_ll_6 = self.cnn_global_features(input_img)
        
        # Classification Network
        
        dense_cl_1 =  Dropout(0.5)(Dense(256, activation="relu", name="dense_cl_1")(dense_gf_2))
        classification_output = Dense(self.n_classes, activation="softmax", name="classification_output")(dense_cl_1)

        # Mid Level Features Network
        conv_ml_2 = self.cnn_mid_features(conv_ll_6)
        
        # Fuse and colorize
        colorization_output = self.cnn_fuse_and_colorize([dense_gf_3, conv_ml_2, conv_ll_2, conv_ll_4, conv_ll_6])
       

        return Model(input_img, [colorization_output, classification_output])
    
    def build_nn_style_transfer(self):
        # for test network we allow arbitray sizes
        
        img_to_colorize = Input(shape=(None, None, 1), name = "img_to_colorize")
        img_with_style = Input(shape=self.img_shape, name = "img_with_style")
        
        # Global Features Network
        dense_gf_3, _,_ = self.cnn_global_features(img_with_style)

        # Mid Level Features Network
        conv_ll_2_img_to_colorize, conv_ll_4_img_to_colorize, conv_ll_6_img_to_colorize = self.cnn_feature_extractor(img_to_colorize)
        conv_ml_2 = self.cnn_mid_features(conv_ll_6_img_to_colorize)
        
        # Fuse and colorize
        colorization_output = self.cnn_fuse_and_colorize([dense_gf_3, conv_ml_2, conv_ll_2_img_to_colorize, conv_ll_4_img_to_colorize, conv_ll_6_img_to_colorize])
       

        return Model([img_to_colorize, img_with_style], colorization_output, name="style_transfer_model")
    
    def build_convolutional_feature_extractor(self):
        input_img = Input(shape=(None,None,1), name = "feature_extractor")
        
        conv_ll_1 = BN()(Conv2D(64, 3, strides = (2,2), padding="same", activation="relu", name="conv_ll_1")(input_img))
        conv_ll_2 = BN()(Conv2D(128, 3, strides = (1,1), padding="same", activation="relu", name="conv_ll_2")(conv_ll_1))
        
        conv_ll_3 = BN()(Conv2D(128, 3, strides = (2,2), padding="same", activation="relu", name="conv_ll_3")(conv_ll_2))
        conv_ll_4 = BN()(Conv2D(256, 3, strides = (1,1), padding="same", activation="relu", name="conv_ll_4")(conv_ll_3))
        
        conv_ll_5 = BN()(Conv2D(256, 3, strides = (2,2), padding="same", activation="relu", name="conv_ll_5")(conv_ll_4))
        conv_ll_6 = BN()(Conv2D(512, 3, strides = (1,1), padding="same", activation="relu", name="conv_ll_6")(conv_ll_5))
        
        self.cnn_feature_extractor = Model(input_img, [conv_ll_2, conv_ll_4,conv_ll_6], name="shared_feature_extractor")
        
    def build_global_features_network(self):
        input_img = Input(shape=self.img_shape, name = "Input_global")
        
#         global_input_img = Reshape(target_shape=self.img_shape)(input_img)
        _, _, global_conv_ll_6 = self.cnn_feature_extractor(input_img)
        
        conv_gf_1 = BN()(Conv2D(512, 3, strides = (2,2), padding="same", activation="relu", name="conv_gf_1")(global_conv_ll_6))
        conv_gf_2 = BN()(Conv2D(512, 3, strides = (1,1), padding="same", activation="relu", name="conv_gf_2")(conv_gf_1))
        
        conv_gf_3 = BN()(Conv2D(512, 3, strides = (2,2), padding="same", activation="relu", name="conv_gf_3")(conv_gf_2))
        conv_gf_4 = BN()(Conv2D(512, 3, strides = (1,1), padding="same", activation="relu", name="conv_gf_4")(conv_gf_3))
        
        flatten_gf = BN()(Flatten()(conv_gf_4))
        
        dense_gf_1 = BN()(Dense(1024, name="dense_gf_1")(flatten_gf))
        dense_gf_2 = BN()(Dense(512, name="dense_gf_2")(dense_gf_1))
        dense_gf_3 = BN()(Dense(256, name="dense_gf_3")(dense_gf_2))
        
        self.cnn_global_features = Model(input_img, [dense_gf_3, dense_gf_2, global_conv_ll_6], name="global_features_extractor")
        
    def build_fuse_and_colorize_network(self):
        dense_gf_3 = Input(shape=(256,), name = "global_features")
        conv_ml_2 = Input(shape=(None, None, 256), name = "mid_level_features")
        conv_ll_2 = Input(shape=(None, None, 128), name = "low_level_features1")
        conv_ll_4 = Input(shape=(None, None, 256), name = "low_level_features2")
        conv_ll_6 = Input(shape=(None, None, 512), name = "low_level_features3")
        
         # Fusion Network
        repeated_global_features = Lambda(self.repeat_vector, output_shape=(None, None, 256)) ([dense_gf_3, conv_ml_2])
        concat_fn = Concatenate(axis=-1, name="concat_fn")([repeated_global_features, conv_ml_2])
        
        # Colorization Network
        
        concat_1 = Concatenate(axis=-1, name="concat_1")([concat_fn, conv_ll_6])
        conv_cl_1 = BN()(Conv2D(128, 3, strides = (1,1), padding="same", activation="relu", name="conv_cl_1")(concat_1))
        upsample_1 = UpSampling2D(name="upsample_1")(conv_cl_1)
        
        concat_2 = Concatenate(axis=-1, name="concat_2")([upsample_1, conv_ll_4])
        conv_cl_2 = BN()(Conv2D(64, 3, strides = (1,1), padding="same", activation="relu", name="conv_cl_2")(concat_2))
        conv_cl_3 = BN()(Conv2D(64, 3, strides = (1,1), padding="same", activation="relu", name="conv_cl_3")(conv_cl_2))
        upsample_2 = UpSampling2D(name="upsample_2")(conv_cl_3) 
        
        concat_3 = Concatenate(axis=-1, name="concat_3")([upsample_2, conv_ll_2])
        conv_cl_4 = BN()(Conv2D(32, 3, strides = (1,1), padding="same", activation="relu", name="conv_cl_4")(concat_3))
        conv_cl_5 = BN()(Conv2D(2, 3, strides = (1,1), activation="sigmoid", padding="same", name="output_cl")(conv_cl_4))
        colorization_output = UpSampling2D(name="colorization_output")(conv_cl_5)
        
        self.cnn_fuse_and_colorize = Model([dense_gf_3, conv_ml_2, conv_ll_2, conv_ll_4, conv_ll_6], colorization_output, name="fuse_and_colorize_model")
        
    def build_mid_level_feature_extractor(self):
        mid_conv_ll_6 = Input(shape=(None, None, 512), name = "shared_features_extracted")
        conv_ml_1 = BN()(Conv2D(512, 3, strides = (1,1), padding="same", activation="relu", name="conv_ml_1")(mid_conv_ll_6))
        conv_ml_2 = BN()(Conv2D(256, 3, strides = (1,1), padding="same", activation="relu", name="conv_ml_2")(conv_ml_1))
        
        self.cnn_mid_features = Model(mid_conv_ll_6, conv_ml_2, name="mid_features_extractor")
        
    def repeat_vector(self, args):
        # Inspiré de https://github.com/keras-team/keras/issues/7949
        
        layer_to_repeat = args[0]
        sequence_layer = args[1]
        res = tile(Reshape((1,1,256))(layer_to_repeat), (1,shape(sequence_layer)[1],shape(sequence_layer)[2],1))
        return res
    
    # def resize_img(self, input_tensor):
    #     input_tensor = args
    
    def load_weights(self, path):
        self.nn.load_weights(path)

    def fit(self, train, val): # train, val DataGenerator

        filepath="weights/weight-%s-{epoch:02d}-{val_loss:.4f}.hdf5"%self.dataset_name
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        
        self.nn.fit_generator(train, epochs = self.epochs, validation_data=val, callbacks = [checkpoint])        
    
    def predict(self, test):
        self.nn.predict(test)
        
    def transfer_style(self, img_to_colorize, img_with_style):
        self.nn_style_transfer([img_to_colorize, img_with_style])
