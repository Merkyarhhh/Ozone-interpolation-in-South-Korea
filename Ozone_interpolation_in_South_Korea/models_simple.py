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

import numpy as np
from sklearn.metrics import mean_squared_error


warnings.filterwarnings('ignore')


class SpatiotemporalMean:
    """
    A model that just returns the mean of all labels
    """
    def __init__(self, use_val=False):
        """
        Initializing the class

        use_val = True means the validation set is used for fitting
        """
        self.use_val = use_val
        prediction_dir = settings.resources_dir + 'models/'
        if use_val:
            prediction_file = 'spatiotemporalmean_predictions_useval.pt'
        else:
            prediction_file = 'spatiotemporalmean_predictions.pt'
        self.prediction_path = prediction_dir + prediction_file

    def __str__(self):
        """
        The name of the Mean simple model.
        """
        return 'SpatiotemporalMean'

    def save_predictions(self):
        """
        Fitting the model. The whole datset is given
        as an argument, since different models need
        different data for fitting.
        """
        print('preparing predictions for SpatiotemporalMean model...')

        # load dataset
        ug = UBAGraph()
        ug.get_dataset()
        dataset = ug[0]

        # find mean
        if self.use_val:
            filter_ = dataset.train_mask | dataset.val_mask
        else:
            filter_ = dataset.train_mask
        mean = torch.mean(dataset.y[filter_])

        # save y hat
        y_hat = torch.full_like(dataset.y, fill_value=mean,
                                dtype=torch.float32)
        torch.save(y_hat, self.prediction_path)
        print(f'written to {self.prediction_path}')

    def read_predictions(self):
        """
        Predictions were saved for future use
        """
        if not os.path.exists(self.prediction_path):
            self.save_predictions()

        self.y_hat = torch.load(self.prediction_path)


class SpatialMean:
    """
    A model that just returns the mean of a specific time step
    over all stations

    If there is no measurement at all, return the mean of
    the CAMS reanalyses of that time step.
    """
    def __init__(self, use_val=False):
        """
        Initializing the class

        use_val = True means the validation set is used for fitting
        """
        self.use_val = use_val
        prediction_dir = settings.resources_dir + 'models/'
        if use_val:
            prediction_file = 'spatialmean_predictions_useval.pt'
        else:
            prediction_file = 'spatialmean_predictions.pt'
        self.prediction_path = prediction_dir + prediction_file

    def __str__(self):
        """
        The name of the SpatialMean simple model.
        """
        return 'SpatialMean'

    def save_predictions(self):
        """
        Fitting the model. The whole datset is given
        as an argument, since different models need
        different data for fitting.
        """
        print('preparing predictions for spatial mean model...')

        # prepare hourly ozone values of shape (n_stations, n_timesteps)
        # (this is the format needed by the pandas imputer)
        hd_o3 = HourlyData('o3')
        hd_o3.read_from_file()
        y_true_df = hd_o3.df.T.copy()
        hd_cams_o3 = HourlyData('cams_o3')
        hd_cams_o3.read_from_file()
        y_cams_df = hd_cams_o3.df.T.copy()

        # UBA Graph data
        ug = UBAGraph()
        ug.get_dataset()
        dataset = ug[0]

        # initialize dataframe containing only Nan
        y_hat_df = pd.DataFrame(index=y_true_df.index,
                                columns=y_true_df.columns)

        # prepare data frames with masks
        n_stations = len(y_true_df.index)
        n_timesteps = len(y_true_df.columns)
        if self.use_val:
            mask = dataset.train_mask | dataset.val_mask
        else:
            mask = dataset.train_mask
        mask_data = mask.reshape((n_stations, n_timesteps)).numpy()
        mask_df = pd.DataFrame(index=y_true_df.index,
                               columns=y_true_df.columns,
                               data=mask_data)

        # the mask is the only data we are allowed to use!
        y_true_df[~mask_df] = np.nan

        # fill missing values, first with mean of true, then of cams
        y_hat_df.fillna(y_true_df.mean(), inplace=True)
        y_hat_df.fillna(y_cams_df.mean(), inplace=True)

        # save
        y_flat_numpy = y_hat_df.values.reshape(-1).astype(np.float32)
        y_flat_pytorch = torch.tensor(y_flat_numpy).view(-1, 1)
        torch.save(y_flat_pytorch, self.prediction_path)
        print(f'written to {self.prediction_path}')

    def read_predictions(self):
        """
        Predictions were saved for future use
        """
        if not os.path.exists(self.prediction_path):
            self.save_predictions()

        self.y_hat = torch.load(self.prediction_path)


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

    def save_predictions(self, limit=settings.limit):
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
        nbrs = NearestNeighbors(n_neighbors=1, n_jobs=7)
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


class RandomForest:
    """
    Training and using random forests.
    """
    def __init__(self, use_val=False):
        """
        Initializing the class

        use_val = True means the validation set is used for fitting
        """
        self.use_val = use_val
        directory = settings.resources_dir + 'models/'
        if use_val:
            prediction_file = 'randomforest_predictions_useval.pt'
            model_file = 'randomforest_model_useval.pkl'
        else:
            prediction_file = 'randomforest_predictions.pt'
            model_file = 'randomforest_model.pkl'
        self.prediction_path = directory + prediction_file
        self.model_path = directory + model_file

    def __str__(self):
        """
        The name of the simple model.
        """
        return 'RandomForest'

    def read_predictions(self):
        """
        Predictions were saved for future use
        """
        if not os.path.exists(self.prediction_path):
            self.save_predictions()

        self.y_hat = torch.load(self.prediction_path)

    def read_model(self):
        """
        Reading the model
        """
        if not os.path.exists(self.model_path):
            self.save_predictions()

        self.model = pickle.load(open(self.model_path, 'rb'))

    def save_predictions(self, features=None, features_not=None, print_=True):
        """
        Train the random forest, then save model and predictions

        features can be a list of features to use
        print_ indicates whether to print info or not
        """
        if print_:
            print('getting random forest predictions (also save model)...')
        # read in data
        ug = UBAGraph()
        x_df = pd.read_csv(ug.x_path, index_col=0)
        y_df = pd.read_csv(ug.y_path, index_col=0)
        mask_df = pd.read_csv(ug.mask_path, index_col=0)

        # use standard features from feature selection if no features given
        if not features:
            features = settings.features
        if not features_not:
            features_not = settings.features_not
        # prepare data
        x_df = x_df[features]
        x_df[features_not] = 0
        x = x_df.values
        if self.use_val:
            mask = mask_df.train_mask | mask_df.val_mask
        else:
            mask = mask_df.train_mask
        x_train = x_df[mask].values
        y_train = y_df[mask].values.reshape(-1)

        # fit model
        rf = RandomForestRegressor(random_state=settings.random_seed,
                                   n_jobs=7, max_depth=15,
                                   n_estimators=500)
        rf.fit(x_train, y_train)

        # save model
        pickle.dump(rf, open(self.model_path, 'wb'))
        if print_:
            print(f'model written to {self.model_path}')

        # save predictions
        y_hat = rf.predict(x)
        y_hat_pytorch = torch.tensor(y_hat.astype(np.float32)).view(-1, 1)
        torch.save(y_hat_pytorch, self.prediction_path)
        if print_:
            print(f'written to {self.prediction_path}')


def tune_nnh():
    """
    tuning the samples parameter of NearestNeighborHybrid
    """
    print('tune samples parameter of nnh model...')

    # UBA Graph data
    ug = UBAGraph()
    ug.get_dataset()
    mask_df = pd.read_csv(ug.mask_path, index_col=0)
    y = ug[0].y.numpy().reshape(-1)[mask_df['val_mask']]

    for limit in range(1, 12):
        nnh = NearestNeighborHybrid()
        nnh.save_predictions(limit=limit)
        nnh.read_predictions()
        y_hat = nnh.y_hat.numpy().reshape(-1)[mask_df['val_mask']]
        rmse = (mean_squared_error(y, y_hat))**.5
        r2 = r2_score(y, y_hat)

        print('\nlimit=', limit)
        print(f'  val R2: {r2:.3f}, val RMSE: {rmse:.3f}')
        print('\n\n\n')


def select_rf_features():
    """
    Choose features for the simple model.
    """
    print('select random forest features by FFS...')

    # UBA Graph data
    ug = UBAGraph()
    ug.get_dataset()

    # read in data
    mask_df = pd.read_csv(ug.mask_path, index_col=0)
    y_val = ug[0].y.numpy().reshape(-1)[mask_df['val_mask']]
    x_df = pd.read_csv(ug.x_path, index_col=0)
    all_features = x_df.columns

    # create pairs
    pair_list = []
    for f1 in all_features:
        for f2 in all_features:
            condition1 = f1 != f2
            condition2 = [f2, f1] not in pair_list
            if (condition1 and condition2):
                pair_list.append([f1, f2])

    # train on all pairs and append to r2 list
    print('\ntesting 2 features:')
    r2_list = []
    for pair in pair_list:
        rf = RandomForest()
        rf.save_predictions(features=pair, print_=False)
        rf.read_predictions()
        y_hat_val = rf.y_hat.numpy().reshape(-1)[mask_df['val_mask']]
        r2 = r2_score(y_val, y_hat_val)
        rmse = (mean_squared_error(y_val, y_hat_val))**.5
        r2_list.append(r2)
        print(f"{', '.join(pair)}: R2 = {r2:.4f}, RMSE = {rmse:.4f}")

    # look for the winning feature
    chosen_features = pair_list[r2_list.index(max(r2_list))]
    best_r2 = max(r2_list)
    print('\nwinners:')
    print(f"{', '.join(chosen_features)}: R2 = {best_r2:.4f}")

    # append new features iteratively until r2 decreases
    for i in range(len(all_features)-2):
        chosen_features_old = chosen_features
        best_r2_old = best_r2
        print(f'\ntesting {len(chosen_features)+1} features:')
        candidates_list = []
        new_r2_list = []
        for feature in all_features:
            if feature in chosen_features_old:
                continue
            candidates = chosen_features + [feature]

            rf = RandomForest()
            rf.save_predictions(features=candidates, print_=False)
            rf.read_predictions()
            y_hat_val = rf.y_hat.numpy().reshape(-1)[mask_df['val_mask']]
            r2 = r2_score(y_val, y_hat_val)
            rmse = (mean_squared_error(y_val, y_hat_val))**.5
            candidates_list.append(candidates)
            new_r2_list.append(r2)
            print(f"{', '.join(candidates)}: R2 = {r2:.4f}, RMSE = {rmse:.4f}")
        chosen_features = candidates_list[new_r2_list.index(
                                             max(new_r2_list))]
        best_r2 = max(new_r2_list)
        print('winners:')
        print(f"{', '.join(chosen_features)}, R2 = {best_r2:.4f}, RMSE = {rmse:.4f}")
        if best_r2 < best_r2_old:
            print('break loop, adding these features harmed generalizability')
            # final result
            print('\nfinal result')
            print(f"{', '.join(chosen_features_old)}, R2 = {best_r2_old:.4f}")
            break

def backward_feature_selection():
    print('select random forest features by BFS...')

    # UBA Graph data
    ug = UBAGraph()
    ug.get_dataset()
    
    # Read in data
    mask_df = pd.read_csv(ug.mask_path, index_col=0)
    y_val = ug[0].y.numpy().reshape(-1)[mask_df['val_mask']]
    x_df = pd.read_csv(ug.x_path, index_col=0)
    all_features = list(x_df.columns)
    
    # Initial training with all features
    print('\nStarting with all features:')
    chosen_features = all_features[:]
    rf = RandomForest()
    rf.save_predictions(features=chosen_features, print_=False)
    rf.read_predictions()
    y_hat_val = rf.y_hat.numpy().reshape(-1)[mask_df['val_mask']]
    best_r2 = r2_score(y_val, y_hat_val)
    best_rmse = (mean_squared_error(y_val, y_hat_val))**.5
    print(f"All features: R2 = {best_r2:.4f}, RMSE = {best_rmse:.4f}")
    
    # Backward elimination: remove features iteratively
    while len(chosen_features) > 1:
        print(f'\nTesting with {len(chosen_features) - 1} features:')
        r2_list = []
        feature_combinations = []
        
        # Test removing each feature one at a time
        for feature in chosen_features:
            temp_features = [f for f in chosen_features if f != feature]
    
            rf = RandomForest()
            rf.save_predictions(features=temp_features, print_=False)
            rf.read_predictions()
            y_hat_val = rf.y_hat.numpy().reshape(-1)[mask_df['val_mask']]
            r2 = r2_score(y_val, y_hat_val)
            rmse = (mean_squared_error(y_val, y_hat_val))**.5
            feature_combinations.append(temp_features)
            r2_list.append(r2)
            print(f"{', '.join(temp_features)}: R2 = {r2:.4f}, RMSE = {rmse:.4f}")
    
        # Find the best combination from this round
        best_index = r2_list.index(max(r2_list))
        best_r2_new = r2_list[best_index]
        best_features_new = feature_combinations[best_index]
        
        # Check if R2 improves after removing the feature
        if best_r2_new < best_r2:
            print('No improvement from removing features.')
            break
    
        # Update chosen_features and best_r2 with the new best set
        chosen_features = best_features_new
        best_r2 = best_r2_new
        print('Best combination so far:')
        print(f"{', '.join(chosen_features)}, R2 = {best_r2:.4f}")
    
    # Final result
    print('\nFinal result:')
    print(f"{', '.join(chosen_features)}, R2 = {best_r2:.4f}")

def backward_feature_importance():
    print('select features importance...')
    
    # UBA Graph data
    ug = UBAGraph()
    ug.get_dataset()
    
    # Read in data
    mask_df = pd.read_csv(ug.mask_path, index_col=0)
    y_val = ug[0].y.numpy().reshape(-1)[mask_df['val_mask']]
    x_df = pd.read_csv(ug.x_path, index_col=0)
    all_features = list(x_df.columns)
    
    # Start with all features
    print('\nStarting with all features:')
    chosen_features = all_features[:]
    rf = RandomForest()
    rf.save_predictions(features=chosen_features, print_=False)
    rf.read_predictions()
    y_hat_val = rf.y_hat.numpy().reshape(-1)[mask_df['val_mask']]
    initial_r2 = r2_score(y_val, y_hat_val)
    initial_rmse = np.sqrt(mean_squared_error(y_val, y_hat_val))
    initial_d = index_of_agreement(y_val, y_hat_val)
    print(f"All features R2 {initial_r2:.4f} RMSE {initial_rmse:.4f} d {initial_d:.4f}")
    
    # Test removing each feature once and calculate metrics
    print('\nTesting by removing one feature at a time:')
    results = []
    for feature in chosen_features:
        temp_features = [f for f in chosen_features if f != feature]
    
        rf = RandomForest()
        rf.save_predictions(features=temp_features, print_=False)
        rf.read_predictions()
        y_hat_val = rf.y_hat.numpy().reshape(-1)[mask_df['val_mask']]
        
        r2 = r2_score(y_val, y_hat_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_hat_val))
        d = index_of_agreement(y_val, y_hat_val)
        
        # Calculate the differences from the initial values
        r2_diff = r2 - initial_r2
        rmse_diff = rmse - initial_rmse
        d_diff = d - initial_d
    
        results.append((feature, r2, r2_diff, rmse, rmse_diff, d, d_diff))
        print(f"Without {feature} R2 {r2:.4f} RMSE {rmse:.4f} d {d:.4f}")
    
    # Final results table
    print('\nResults of removing each feature once:')
    for feature, r2, r2_diff, rmse, rmse_diff, d, d_diff in results:
        print(f"Without {feature} R2 {r2_diff:.4f} RMSE {rmse_diff:.4f} d {d_diff:.4f}")

def evaluate_models():
    """
    Evaluate all simple models
    """
    print('evaluate simple models...')
    # settings for results
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    # read graph dataset
    ug = UBAGraph()
    ug.get_dataset()
    dataset = ug[0]
    y = dataset.y.numpy().reshape(-1)

    # which models to evaluate
    models = [SpatiotemporalMean, SpatialMean,
              NearestNeighborHybrid, RandomForest]
    use_vals = [True]

    specs = [(model, use_val) for model in models for use_val in use_vals]

    # evaluation
    for model, use_val in specs:
        m = model(use_val=use_val)
        m.read_predictions()
        y_hat = m.y_hat.numpy().reshape(-1)
        if use_val:
            train_mask = dataset.train_mask.numpy().reshape(-1) | \
                         dataset.val_mask.numpy().reshape(-1)
            test_mask = dataset.test_mask.numpy().reshape(-1)
            train_mask_name = 'train+val'
            test_mask_name = 'test'
        else:
            train_mask = dataset.train_mask.numpy().reshape(-1)
            test_mask = dataset.val_mask.numpy().reshape(-1)
            train_mask_name = 'train'
            test_mask_name = 'val'
        train_r2 = r2_score(y[train_mask], y_hat[train_mask])
        train_rmse = (mean_squared_error(y[train_mask], y_hat[train_mask]))**.5
        train_d = index_of_agreement(y[train_mask], y_hat[train_mask])
        test_r2 = r2_score(y[test_mask], y_hat[test_mask])
        test_rmse = (mean_squared_error(y[test_mask], y_hat[test_mask]))**.5
        test_d = index_of_agreement(y[test_mask], y_hat[test_mask])
        #print(f'\n{m}, use_val={use_val}')
        #print(f'evaluation on {train_mask_name} set(s)')
        #print(f'R2: {train_r2:.3f} , RMSE: {train_rmse:.3f} , d: {train_d:.3f}')
        #print(f'evaluation on {test_mask_name} set')
        print(f'R2: {test_r2:.3f} , RMSE: {test_rmse:.3f} , d: {test_d:.3f}')
        gle = GapLengthEvaluation()
        #print(f'bin evaluation on {test_mask_name}')
        print(gle.evaluate_bins_separately(test_mask_name+'_mask', y, y_hat))
        print('-'*20)


if __name__ == '__main__':
    """
    Tune the simple models, then save their predictions.
    Also save the predictions where the validation set is used
    for training.
    """
    #model = SpatiotemporalMean()
    #model.save_predictions()
    #model = SpatiotemporalMean(use_val=True)
    #model.read_predictions()
    #model.save_predictions()

    #model = SpatialMean()
    #model.save_predictions()
    #model = SpatialMean(use_val=True)
    #model.save_predictions()

    #tune_nnh()
    #model = NearestNeighborHybrid()
    #model.save_predictions()
    #model = NearestNeighborHybrid(use_val=True)
    #model.save_predictions()

    #select_rf_features()
    #backward_feature_selection()
    #backward_feature_importance()
    model = RandomForest()
    model.save_predictions()
    model = RandomForest(use_val=True)
    model.read_predictions()
    model.save_predictions()

    #evaluate_models()
