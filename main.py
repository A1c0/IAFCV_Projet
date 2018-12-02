from src.DataManager_td import DataManager
from src.NeuralNetwork_td import NeuralNetwork
import numpy as np


def main():
    data_manager = DataManager()
    data_manager.loadData()
    neural_network = NeuralNetwork()
    neural_network.create_model()
    neural_network.train(data_manager.train_data, data_manager.train_labels, data_manager.eval_data,
                         data_manager.eval_labels, epochs=100)
    result = neural_network.evaluate(data_manager.eval_data, data_manager.eval_labels)
    print("Accuracy : {}".format(result))
    pass


if __name__ == "__main__":
    main()
