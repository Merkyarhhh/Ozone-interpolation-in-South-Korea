"""
Training a pytorch geometric graph attention network on AQ-Bench
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
from models_modelwrapper import ModelWrapper
from models_graph_attention_network import GAT


if __name__ == '__main__':
    """
    Set up model and train it
    """
    gaqb = ModelWrapper()
    gaqb.get_data(AQBenchGraph)
    gaqb.get_model(GAT)
    gaqb.training(learning_rate=0.01, n_epoch=2000)
    with torch.no_grad():
        y_hat = gaqb.model(gaqb.data.x, gaqb.data.edge_index,
                           gaqb.data.edge_attr)
        y_test = gaqb.data.y[~gaqb.data.train_mask]
        y_test_hat = y_hat[~gaqb.data.train_mask]
        RMSE = mean_squared_error(y_test, y_test_hat)**.5
        R2 = r2_score(y_test, y_test_hat)
        print('======================')
        print(f'Final evaluation on test set')
        print(f'RMSE: {RMSE:.2f}; R2: {R2:.2f}')
        plt.scatter(y_hat, gaqb.data.y)
        plt.xlim([0, 70])
        plt.ylim([0, 70])
        plt.show()
