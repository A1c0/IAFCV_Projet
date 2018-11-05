import numpy as np
from tensorflow.keras.datasets import cifar10

class DataManager(object):

    def __init__(self):
        self.train_data = None
        self.train_labels = None
        self.eval_data = None
        self.eval_labels = None

    
    def loadData(self):
        """Load the data from cifar-10-batches.
           See http://www.cs.toronto.edu/~kriz/cifar.html for instructions on
           how to do so.
        """
        (self.train_data, self.train_labels), (self.eval_data, self.eval_labels) = cifar10.load_data()
        self.train_data /= 255
        self.eval_data /= 255
        pass
