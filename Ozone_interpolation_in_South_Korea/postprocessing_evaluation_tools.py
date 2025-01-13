"""
Tools for evaluation of our different models
"""

# general
import pdb
import warnings
from os.path import exists

# data science
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors

# pytorch
import torch

# own package
import settings
from retrieval_toar_db import StationData
from preprocessing_uba import UBAGraph
from preprocessing_utils import geo_to_cartesian
from postprocessing_utils import confidence_interval


# oppress future warnings
warnings.filterwarnings('ignore')


def print_best_hyperparameters():
    """
    A small helper function that will read in the tuning file
    and print the best hyper parameters for each model.
    """
    print('Printing best hyperparameters...')

    # read in
    df = pd.read_csv(settings.output_dir+'tuning.csv', index_col=0)

    # find hyperparams
    simple_models = list(np.unique(df.simple_model))

    for simple_model in simple_models:
        df_filtered = df[df.simple_model == simple_model]
        print(len(df_filtered))
        max_r2 = max(df_filtered.r2)
        df_best = df_filtered[df_filtered.r2==max_r2]
        index_best = df_best.index[0]

        # print results
        print()
        for col in df.columns:
            value = df.at[index_best, col]
            print(f'{col}: {value}')
        print(f'found at index {index_best}')


def rmse(y, y_hat):
    """
    The root mean square error.
    """

    rmse = (mean_squared_error(y, y_hat))**.5

    return rmse


def index_of_agreement(y, y_hat):
    """
    Willmott's index of agreement
    """
    y = np.array(y)
    y_hat = np.array(y_hat)

    a = np.sum((y-y_hat)**2)
    b = np.sum((np.abs(y_hat-np.mean(y)) + np.abs(y-np.mean(y)))**2)
    d = 1 - a/b

    return d


class GapLengthEvaluation:
    """
    For missing data imputation it is important
    to evaluate the models separately for different
    gap lengths.
    This class prepares the necessary files
    and evaluation routines to be imported from
    experiment workflows.
    """
    def __init__(self):
        """
        Initialize the class
        """
        # UBA graph has all data paths needed
        self.ug = UBAGraph()
        self.gap_df = pd.read_csv(self.ug.gap_path, index_col=0)
        self.mask_df = pd.read_csv(self.ug.mask_path, index_col=0)

        # bins
        self.bins = settings.bins

    def print_existing_gaps(self):
        """
        This routine prints info about gaps that occur or that
        were flagged.
        """
        print('printing gaps gaps...')

        # data
        gap_df = self.gap_df
        mask_df = self.mask_df
        bins = self.bins

        # print gap info
        for type_ in ['missing_o3_mask', 'val_mask', 'test_mask']:
            print(f'\nanalyzing {type_}')
            gap_df_type = gap_df[gap_df.type==type_]
            mask_df_type = mask_df[type_]

            for correlated in [False]:
                for idx, bin_ in enumerate(bins[:-1]):
                    lower_border = bin_
                    upper_border = bins[idx+1] - 1

                    gap_df_filtered = gap_df_type[
                                   (gap_df_type.correlated==correlated)
                                 & (gap_df_type.len>=lower_border)
                                 & (gap_df_type.len<=upper_border)]

                    n_gaps = len(gap_df_filtered)
                    n_samples = int(gap_df_filtered.n_samples.sum())

                    print(#f'correlated: {correlated} ' +
                          #f'lowerborder: {lower_border} h ' +
                          #f'upperborder: {upper_border} h ' +
                          f'{n_gaps} ' +
                          f'{n_samples}')

            print(f' {int(gap_df_type.n_samples.sum())}')
            print(f'totaln_samplesaccordingtomask: {mask_df_type.sum()}')

    def evaluate_bins_separately(self, type_, y, y_hat):
        """
        Separate evaluation for all bins

        type_ is one of, 'val_mask', 'test_mask'
        y and y_hat are pytorch tensors or numpy arrays
        """
        # data
        gap_df = self.gap_df
        mask_df = self.mask_df
        bins = self.bins

        # initialize
        columns = ['correlated', 'lower_border', 'upper_border',
                   'n_gaps', 'n_samples', 'r2', 'rmse', 'd']
        bin_evaluation_df = pd.DataFrame(index=range(len(self.bins)-1),
                            columns=columns)

        # prepare data
        if not isinstance(y, np.ndarray):
            y = y.numpy()
        if not isinstance(y_hat, np.ndarray):
            y_hat = y_hat.numpy()
        mask_index_list = mask_df[mask_df[type_]].index.to_list()

        # fill df
        counter = 0
        for correlated in [False, True]:
            for bin_index, lower_border in enumerate(self.bins[:-1]):
                upper_border = self.bins[bin_index+1] - 1

                gap_df_filtered = gap_df[
                               (gap_df.type==type_)
                             & (gap_df.correlated==correlated)
                             & (gap_df.len>=lower_border)
                             & (gap_df.len<=upper_border)]

                gap_index_list = []
                start_idx = gap_df_filtered.start_idx
                end_idx = start_idx + gap_df_filtered.len
                zip_ = list(zip(start_idx, end_idx))
                for start_idx, end_idx in zip_:
                    gap_index_list.extend(range(start_idx, end_idx))

                index_list = list(set(mask_index_list) & set(gap_index_list))

                y_filtered = y[index_list]
                y_hat_filtered = y_hat[index_list]
                n_gaps = len(gap_df_filtered)
                n_samples = len(y_filtered)

                if n_samples == 0:
                    continue
                r2 = r2_score(y_filtered, y_hat_filtered)
                rmse = (mean_squared_error(y_filtered, y_hat_filtered))**.5
                d = index_of_agreement(y_filtered, y_hat_filtered)
                _ = [correlated, lower_border, upper_border,
                     n_gaps, n_samples, r2, rmse, d]
                bin_evaluation_df.loc[counter] = _
                counter += 1

        return bin_evaluation_df


class Bootstrap():
    """
    A class for bootstrapping evaluation metrics
    """
    def __init__(self, metric, n):
        """
        Initialize the class with the evaluation metric and the
        number of bootstraps
        """
        self.metric = metric
        self.n = n

    def bootstrap(self, y, y_hat):
        """
        Bootstrap the given samples.
        """
        # check inputs
        y = np.array(y).reshape(-1)
        y_hat = np.array(y_hat).reshape(-1)

        # call bootstrap procedure
        n_data_samples = len(y)
        data = np.arange(n_data_samples)
        def statistic(idxs):
            y_resampled = y[idxs]
            # print(f'y resampled: {y_resampled}')
            y_hat_resampled = y_hat[idxs]
            # print(f'y_hat_resampled: {y_hat_resampled}')
            return self.metric(y_resampled, y_hat_resampled)
        np.random.seed(settings.random_seed)

        metrics = []

        for _ in range(self.n):
            x = np.random.choice(data, size=n_data_samples, replace=True)
            # print(x)
            metric = statistic(x)
            metrics.append(metric)
            print(metric)
            # print()

        print(confidence_interval(metrics))


def count_exceedances(print_=False):
    """
    Count exceedances of information and warning thresholds
    in imputed and unimputed dataset

    NB for this, a final imputed dataset using the best methods
    needs to be created.
    """
    if print_:
        print('counting exceedances...')

    # load data
    ug = UBAGraph()
    y_true_df = pd.read_csv(ug.y_path, index_col=0)
    y_true = y_true_df.y.to_numpy().reshape(-1)
    imputed_path = settings.resources_dir + \
                   'imputed_dataset/imputed_o3.csv'
    y_imp_df = pd.read_csv(imputed_path, index_col=0)
    y_imp = y_imp_df.values.reshape(-1)

    # set up data frame
    columns = ['threshold', 'n_measured', 'n_imputed',
               'difference']
    exc_df = pd.DataFrame(columns=columns)

    # calculate exceedances
    thresholds = list(range(50, 121, 5))
    for idx, threshold in enumerate(thresholds):
        tru_exc = (y_true>=threshold).sum()
        imp_exc = (y_imp>=threshold).sum()
        diff = imp_exc - tru_exc

        _ = [threshold, tru_exc, imp_exc, diff]
        exc_df.at[idx] = _

        if not print_:
            continue
        print(f'\nthreshold: {threshold}')
        print(f'measured exc: {tru_exc}, imputed exc: {diff}')
        print('after imputation:', tru_exc+diff)

    return exc_df


def number_of_neighbors(station_id, return_ids=False):
    """
    How many stations are in a radius of 50 km?
    """
    # read in data
    sd = StationData()
    sd.read_from_file()

    # prepare
    sd.df['idx'] = range(len(sd.df))
    station_idx = sd.df.at[station_id, 'idx']
    max_dist = 8
    x_, y_, z_ = geo_to_cartesian(sd.df.lon.values,
                                  sd.df.lat.values)
    coords = np.array([x_, y_, z_]).T
    station_coords = coords[station_idx, :].reshape((1, 3))

    # find neighbors
    nn = NearestNeighbors()
    nn.fit(coords)
    dist_lists, idx_lists = nn.radius_neighbors(station_coords,
                                                radius=max_dist)
    n_neighbors = len(idx_lists[0]) - 1
    distances = sorted(dist_lists[0])[1:]
    ids = sd.df.index[idx_lists[0]].to_list()[1:]

    # return
    if return_ids:
        return(n_neighbors, distances, ids)
    return(n_neighbors, distances)


def get_characteristics_df(how='read'):
    """
    A dataframe where we preprocess some gap characteristics for
    easier plotting
    """
    # read in data
    ug = UBAGraph()
    y_true_df = pd.read_csv(ug.y_path, index_col=0)
    gap_df = pd.read_csv(ug.gap_path, index_col=0)
    mask_df = pd.read_csv(ug.mask_path, index_col=0)
    reg_df = pd.read_csv(ug.reg_path, index_col=0)
    model_path = settings.resources_dir + 'models/'
    stm_file = 'spatiotemporalmean_predictions_useval.pt'
    sm_file = 'spatialmean_predictions_useval.pt'
    nnh_file = 'nearestneighborhybrid_predictions_useval.pt'
    rf_file = 'randomforest_predictions_useval.pt'
    stm_cs_file = 'SpatiotemporalMean_cs_predictions_useval.pyt'
    sm_cs_file = 'SpatialMean_cs_predictions_useval.pyt'
    nnh_cs_file = 'NearestNeighborHybrid_cs_predictions_useval.pyt'
    rf_cs_file = 'RandomForest_cs_predictions_useval.pyt'
    
    stm_imp = torch.load(model_path+stm_file).numpy().reshape(-1)
    sm_imp = torch.load(model_path+sm_file).numpy().reshape(-1)
    nnh_imp = torch.load(model_path+nnh_file).numpy().reshape(-1)
    rf_imp = torch.load(model_path+rf_file).numpy().reshape(-1)
    stm_cs_imp = torch.load(model_path+stm_cs_file).numpy().reshape(-1)
    sm_cs_imp = torch.load(model_path+sm_cs_file).numpy().reshape(-1)
    nnh_cs_imp = torch.load(model_path+nnh_cs_file).numpy().reshape(-1)
    rf_cs_imp = torch.load(model_path+rf_cs_file).numpy().reshape(-1)

    # path
    save_path = settings.output_dir + 'characteristics_tmp.csv'

    # set up and fill df
    if how == 'prepare':
        # set up df
        columns = ['y_true', 'y_imputed',
                   'y_stm', 'y_sm', 'y_nnh', 'y_rf',
                   'y_stm_cs', 'y_sm_cs', 'y_nnh_cs', 'y_rf_cs',
                   'gap_len', 'correlated',
                   'station_id', 'n_neighbors', 'test_mask']
        df = pd.DataFrame(columns=columns, index=y_true_df.index)

        # columns that we can take from other data frames
        df.y_true = y_true_df.y
        df.y_stm = stm_imp
        df.y_sm = sm_imp
        df.y_nnh = nnh_imp
        df.y_rf = rf_imp
        df.y_stm_cs = stm_cs_imp
        df.y_sm_cs = sm_cs_imp
        df.y_nnh_cs = nnh_cs_imp
        df.y_rf_cs = rf_cs_imp
        df.station_id = reg_df.station_id
        df.test_mask = mask_df.test_mask

        # number of neighbors of each station
        station_ids = np.unique(df.station_id)
        for station_id in station_ids:
            n_neighbors, _ = number_of_neighbors(station_id)
            df.loc[df.station_id==station_id, 'n_neighbors'] = \
                                                            n_neighbors

        # a df of all test gaps, the only gaps we analyze here
        gap_df_filtered = gap_df[gap_df.type=='test_mask']
        gap_df_filtered.reset_index(inplace=True)
        for gap_idx, gap_row in gap_df_filtered.iterrows():
            if gap_idx % 1000 == 0:
                print(f'{gap_idx/len(gap_df_filtered)*100:.2f} %')
                df.to_csv(save_path)
            start_idx = gap_row.start_idx
            end_idx = gap_row.start_idx + gap_row.len - 1
            len_ = gap_row.len
            correlated = gap_row.correlated

            df.loc[start_idx:end_idx, 'gap_len'] = len_
            df.loc[start_idx:end_idx, 'correlated'] = correlated
            
            
            if len_ == 1:
                df.loc[start_idx:end_idx, 'y_imputed'] = \
                                           nnh_imp[start_idx:end_idx+1]
            elif len_ == 2:
                df.loc[start_idx:end_idx, 'y_imputed'] = \
                                           nnh_cs_imp[start_idx:end_idx+1]
            elif len_ == 3:
                df.loc[start_idx:end_idx, 'y_imputed'] = \
                                           rf_cs_imp[start_idx:end_idx+1]
            elif len_ == 4:
                df.loc[start_idx:end_idx, 'y_imputed'] = \
                                           rf_cs_imp[start_idx:end_idx+1]
            elif len_ == 5:
                df.loc[start_idx:end_idx, 'y_imputed'] = \
                                           rf_cs_imp[start_idx:end_idx+1]
            else:
                df.loc[start_idx:end_idx, 'y_imputed'] = \
                                           rf_cs_imp[start_idx:end_idx+1]
        df.to_csv(save_path)

    # if it was already prepared
    elif how == 'read':
        df = pd.read_csv(save_path, index_col=0)

    # return
    return df


if __name__ == '__main__':
    """
    Start or test routines.
    """
    index_of_agreement_ = False
    gap_length_evaluation_ = True
    print_best_hyperparameters_ = False
    test_bootstrap_ = False
    count_exceedances_ = False
    number_of_neighbors_ = False
    get_characteristics_df_ = True

    if index_of_agreement_:
        # test index of agreement should be.959
        y = [1, 3, 5, 3, 9]
        y_hat = [2, 4, 6, 2, 8]

        print(f'index of agreement: {index_of_agreement(y, y_hat):.3f}')

    if gap_length_evaluation_:
        # test bin evaluation
        gle = GapLengthEvaluation()
        gle.print_existing_gaps()

        from models_simple import RandomForest
        gle.ug.get_dataset()
        dataset = gle.ug
        data = dataset[0]
        rf = RandomForest()
        rf.read_predictions()
        y = data.y
        y_hat = rf.y_hat
        bin_evaluation_df = gle.evaluate_bins_separately('val_mask', y, y_hat)
        print(bin_evaluation_df)

    if print_best_hyperparameters_:
        print_best_hyperparameters()

    if test_bootstrap_:
        # test bootstrap
        mask = np.array(data.val_mask).reshape(-1)
        b = Bootstrap(metric=index_of_agreement, n=1000)
        b.bootstrap(y[mask], y_hat[mask])

    if count_exceedances_:
        count_exceedances(print_=True)

    if number_of_neighbors_:
        n, dists = number_of_neighbors(3443)
        print(n)
        print(dists)

    if get_characteristics_df_:
        # df = get_characteristics_df(how='read')
        df = get_characteristics_df(how='prepare')

