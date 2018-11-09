import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np


class NeuralNetwork(object):

    def __init__(self):
        self.model = None

    def create_model(self):
        """Create and compile the keras model. See layers-18pct.cfg and 
           layers-params-18pct.cfg for the network model, 
           and https://code.google.com/archive/p/cuda-convnet/wikis/LayerParams.wiki 
           for documentation on the layer format.
        """
        self.model = keras.models.Sequential()
        self.model.add(
            keras.layers.Conv2D(input_shape=(32, 32, 3), filters=32, kernel_size=5, strides=(1, 1), padding='same', data_format='channels_last',
                                dilation_rate=(1, 1), activation=None, use_bias=True,
                                kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None,
                                bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                                bias_constraint=None))  # CONV1

        self.model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding='valid', data_format='channels_last'))
        self.model.add(keras.layers.BatchNormalization())
        self.model.add(
            keras.layers.Conv2D(filters=32, kernel_size=5, strides=(1, 1), padding='valid', data_format='channels_last',
                                dilation_rate=(1, 1), activation=None, use_bias=True,
                                kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None,
                                bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                                bias_constraint=None))  # CONV2

    def train(self, train_data, train_labels, epochs):
        """Train the keras model
        
        Arguments:
            train_data {np.array} -- The training image data
            train_labels {np.array} -- The training labels
            epochs {int} -- The number of epochs to train for
        """

        pass

    def evaluate(self, eval_data, eval_labels):
        """Calculate the accuracy of the model
        
        Arguments:
            eval_data {np.array} -- The evaluation images
            eval_labels {np.array} -- The labels for the evaluation images
        """
        pass

    def test(self, test_data):
        """Make predictions for a list of images and display the results
        
        Arguments:
            test_data {np.array} -- The test images
        """
        pass

    # Exercise 7 Save and load a model using the keras.models API
    def save_model(self, save_file="model.h5"):
        """Save a model using the keras.models API
        
        Keyword Arguments:
            saveFile {str} -- The name of the model file (default: {"model.h5"})
        """
        pass

    def load_model(self, save_file="model.h5"):
        """Load a model using the keras.models API
        
        Keyword Arguments:
            saveFile {str} -- The name of the model file (default: {"model.h5"})
        """
        pass