"""
Training a basic pytorch model on AQ-Bench
"""

# general
import pdb

# data science
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# pytorch
import torch

# plotting
import matplotlib.pyplot as plt

# own package
from preprocessing_aqbench import AQBenchGraph
from models_neural_network import NN
from models_modelwrapper import ModelWrapper


if __name__ == '__main__':
    """
    Set up model and train it
    """
    nnaqb = ModelWrapper()
    nnaqb.get_data(AQBenchGraph)
    nnaqb.get_model(NN)
    nnaqb.training(learning_rate=0.01, n_epoch=2000)
    with torch.no_grad():
        y_hat = nnaqb.model(nnaqb.data.x, nnaqb.data.edge_index,
                           nnaqb.data.edge_attr)
        y_test = nnaqb.data.y[~nnaqb.data.train_mask]
        y_test_hat = y_hat[~nnaqb.data.train_mask]
        RMSE = mean_squared_error(y_test, y_test_hat)**.5
        R2 = r2_score(y_test, y_test_hat)
        print('======================')
        print(f'Final evaluation on test set')
        print(f'RMSE: {RMSE:.2f}; R2: {R2:.2f}')
        plt.scatter(y_hat, nnaqb.data.y)
        plt.xlim([0, 70])
        plt.ylim([0, 70])
        plt.show()
