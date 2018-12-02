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
            keras.layers.Conv2D(input_shape=(32, 32, 3), filters=32, kernel_size=5, padding='same',
                                data_format='channels_last', activation=tf.nn.relu))  # CONV1
        self.model.add(
            keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'))  # POOL1
        self.model.add(keras.layers.Dropout(0.5))
        self.model.add(keras.layers.BatchNormalization())  # RNORM1
        self.model.add(
            keras.layers.Conv2D(filters=64, kernel_size=5, padding='same', data_format='channels_last'))  # CONV2
        self.model.add(
            keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same',
                                   data_format='channels_last'))  # POOL2
        self.model.add(keras.layers.BatchNormalization())  # RNORM2
        self.model.add(
            keras.layers.Conv2D(filters=64, kernel_size=5, padding='same', data_format='channels_last'))  # CONV3
        self.model.add(
            keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same',
                                   data_format='channels_last'))  # POOL3
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(10, activation=tf.nn.softmax))  # FC10

        self.model.compile(optimizer=keras.optimizers.Adam(),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, train_data, train_labels, eval_data, eval_labels, epochs):
        """Train the keras model
        
        Arguments:
            train_data {np.array} -- The training image data
            train_labels {np.array} -- The training labels
            epochs {int} -- The number of epochs to train for
        """
        self.model.fit(train_data, train_labels, epochs=epochs, validation_data=(eval_data, eval_labels))

    def evaluate(self, eval_data, eval_labels):
        """Calculate the accuracy of the model
        
        Arguments:
            eval_data {np.array} -- The evaluation images
            eval_labels {np.array} -- The labels for the evaluation images
        """
        return self.model.evaluate(eval_data, eval_labels)[1]

    def test(self, test_data):
        """Make predictions for a list of images and display the results
        
        Arguments:
            test_data {np.array} -- The test images
        """
        return self.model.predict(test_data)

    # Exercise 7 Save and load a model using the keras.models API
    def save_model(self, save_file="model.h5"):
        """Save a model using the keras.models API
        
        Keyword Arguments:
            saveFile {str} -- The name of the model file (default: {"model.h5"})
        """
        self.model.save(save_file)

    def load_model(self, save_file="model.h5"):
        """Load a model using the keras.models API
        
        Keyword Arguments:
            saveFile {str} -- The name of the model file (default: {"model.h5"})
        """
        self.model = keras.models.load_model(save_file)
