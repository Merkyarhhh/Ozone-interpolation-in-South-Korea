"""
Create the final dataset. It contains ozone measurements wherever
they are available. According to the results of this study,
gaps of up to 5 h lenght are linearly interpolated, while gaps of 6 h
or more are imputed with random forest + correct and smooth.
"""

# general
import pdb

# data science
import numpy as np
import pandas as pd

# pytorch
import torch

# own package
import settings
from preprocessing_uba import UBAGraph


def create_final_imputation():
    """
    Create final imputed dataset using the best methods:

    Linear interpolation for short gaps, RF + C&S for long gaps.
    """
    print('produce final imputed dataset...')

    # load data

    gap_df = pd.read_csv("C:/Users/sjjun/OneDrive/Desktop/data_seoul/uba_graph_preproc/gap.csv", index_col=0)
    mask_df = pd.read_csv("C:/Users/sjjun/OneDrive/Desktop/data_seoul/uba_graph_preproc/mask.csv", index_col=0)
    missing_o3_mask = mask_df.missing_o3_mask.to_numpy().reshape(-1)
    y_true_df = pd.read_csv("C:/Users/sjjun/OneDrive/Desktop/data_seoul/uba_graph_preproc/y.csv", index_col=0)
    y_true = y_true_df.y.to_numpy().reshape(-1)
    reg_df = pd.read_csv("C:/Users/sjjun/OneDrive/Desktop/data_seoul/uba_graph_preproc/reg.csv", index_col=0)

    # load imputations of the models
    lin_imp = torch.load("C:/Users/sjjun/OneDrive/Desktop/data_seoul/models/nearestneighborhybrid_predictions_useval.pt").numpy().reshape(-1)
    rf_cs_imp = torch.load("C:/Users/sjjun/OneDrive/Desktop/data_seoul/models/randomForest_cs_predictions_useval.pyt").numpy().reshape(-1)

    # write imputations to gaps
    imp = y_true.copy()
    imp[missing_o3_mask] = rf_cs_imp[missing_o3_mask]
    gap_df_filtered = gap_df[(gap_df.type=='missing_o3_mask') &
                             (gap_df.len<2)]
    for gap_idx, gap_row in gap_df_filtered.iterrows():
        for idx in range(gap_row.start_idx, gap_row.start_idx+gap_row.len):
            if missing_o3_mask[idx]:
                imp[idx] = lin_imp[idx]
    imp_df = pd.DataFrame(index=y_true_df.index,
                          columns=['y_imputed'],
                          data=imp)
    imp_df.index.name = 'node_index'

    # prepare data for saving
    station_list = np.unique(reg_df.station_id)
    n_stations = len(station_list)
    datetime_list = np.unique(reg_df.datetime)
    n_datetime = len(datetime_list)
    imp_o3_df = pd.DataFrame(index=datetime_list,
                             columns=station_list,
                             data=imp.reshape(n_stations, n_datetime).T)
    imp_o3_df.index.name = 'datetime'

    # prepare an info dataframe
    imp_info_df = pd.DataFrame(index=datetime_list,
                               columns=station_list,
                               data=missing_o3_mask.reshape(n_stations,
                                                            n_datetime).T)
    imp_info_df.index.name = 'datetime'

    # save data
    imp_o3_df.to_csv("C:/Users/sjjun/OneDrive/Desktop/data_seoul/imputed_dataset/imputed_o3.csv")
    imp_info_df.to_csv("C:/Users/sjjun/OneDrive/Desktop/data_seoul/imputed_dataset/imputed_info.csv")


if __name__ == '__main__':
    """
    Start or test routines.
    """
    create_final_imputation_ = True

    if create_final_imputation_:
        create_final_imputation()

