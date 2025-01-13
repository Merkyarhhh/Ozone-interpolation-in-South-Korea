"""
This script contains routines to fit and evaluate simple models as
baselines and to combine with correct and smooth.
"""

# general
import os
import random
import pdb
import warnings
import pickle
import datetime

# data science
import numpy as np
import pandas as pd

# pytorch
import torch

# sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors

# own package
import settings
from retrieval_toar_db import HourlyData
from preprocessing_uba import UBAGraph
from preprocessing_utils import get_one_hot
from postprocessing_evaluation_tools import index_of_agreement
from postprocessing_evaluation_tools import GapLengthEvaluation



class NearestNeighborHybrid:
    """
    A model that performs simple imputation
    as Junninen et al. (2004).
    Linear interpolation in time dimension
    (for short gaps)
    Nearest neighbors in feature dimension.
    (for longer gaps)
    """
    def __init__(self, use_val=False):
        """
        Initializing the class

        use_val = True means the validation set is used for fitting
        """
        self.use_val = use_val
        prediction_dir = settings.resources_dir + 'models/'
        if use_val:
            prediction_file = 'nearestneighborhybrid_predictions_useval.pt'
        else:
            prediction_file = 'nearestneighborhybrid_predictions.pt'
        self.prediction_path = prediction_dir + prediction_file

    def __str__(self):
        """
        The name of the Nearest Neighbors simple model.
        """
        return 'NearestNeighborHybrid'

    def read_predictions(self):
        """
        Predictions were saved for future use
        """
        if not os.path.exists(self.prediction_path):
            self.save_predictions()

        self.y_hat = torch.load(self.prediction_path)

    def save_predictions(self, limit=5):
        """
        Fitting the model. The whole datset is given
        as an argument, since different models need
        different data for fitting.
        """
        print('getting nearest neighbor hybrid predictions...')

        # prepare hourly true ozone values of shape (n_timesteps, n_stations)
        hd = HourlyData('o3')
        hd.read_from_file()
        y_df = hd.df.copy()

        # UBA Graph data
        ug = UBAGraph()
        ug.get_dataset()

        # set test or val and test samples to nan
        mask_flat_df = pd.read_csv(ug.mask_path, index_col=0)
        if self.use_val:
            mask_list = ['test_mask']
        else:
            mask_list = ['val_mask', 'test_mask']
        for mask in mask_list:
            mask_data = mask_flat_df[mask].values.reshape(len(y_df.columns),
                                                         (len(y_df.index))).T
            mask_df = pd.DataFrame(columns=y_df.columns,
                                   index=y_df.index, data=mask_data)
            y_df[mask_df] = np.nan

        # linear interpolation for short gaps
        # n.b. we only want to fill gaps that are surrounded by
        # true values, this is why we have to compare forward
        # and backward imputation
        y_df_before_linear = y_df.copy()
        y_df_forward = y_df.interpolate(axis=0, limit=limit, limit_direction='forward')
        y_df_backward = y_df.interpolate(axis=0, limit=limit, limit_direction='backward')
        y_df_both = y_df.interpolate(axis=0, limit=limit, limit_direction='both')
        lin_imp_mask_df = np.isnan(y_df_before_linear) & \
                     ~np.isnan(y_df_forward) & ~np.isnan(y_df_backward)
        y_flat_both = y_df_both.values.T.reshape(-1)
        lin_imp_filter = lin_imp_mask_df.values.T.reshape(-1)

        # prepare x and y for neighbor search
        y_numpy_flat = y_df.values.T.reshape(-1)
        x_numpy = ug[0].x.numpy()
        scaler = StandardScaler()
        x_numpy = scaler.fit_transform(x_numpy)

        # find neighbors
        if self.use_val:
            training_index = mask_flat_df.train_mask | mask_flat_df.val_mask
        else:
            training_index = mask_flat_df.train_mask
        nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1)
        nbrs.fit(x_numpy[training_index])
        distances, indices = nbrs.kneighbors(x_numpy)
        y_training = y_numpy_flat[training_index]  # same index!
        y_hat_numpy = np.array([y_training[idx][0] for idx in indices])

        # for short gaps: take linear interpolation!
        y_hat_numpy[lin_imp_filter] = y_flat_both[lin_imp_filter]

        # save
        y_hat_pytorch = torch.tensor(y_hat_numpy, dtype=torch.float32).view(-1, 1)
        torch.save(y_hat_pytorch, self.prediction_path)
        print(f'written to {self.prediction_path}')

model = NearestNeighborHybrid()  # 인스턴스 생성
model.save_predictions()  # read_predictions 메서드 호출
