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


def create_final_imputation():
    """
    Create final imputed dataset using the best methods:

    Linear interpolation for short gaps, RF + C&S for long gaps.
    """
    print('produce final imputed dataset...')

    # load data
    gap_df = pd.read_csv("C:/Users/sjjun/OneDrive/Desktop/spatial-patterns-data/uba_graph_preproc/gap.csv") ###모든 결측치 정보###
    missing_o3_mask = pd.read_csv("C:/Users/sjjun/OneDrive/Desktop/spatial-patterns-data/uba_graph_preproc/mask.csv") ###확인필요, 결측치:TRUE else FALSE###
    y_true = pd.read_csv("C:/Users/sjjun/OneDrive/Desktop/spatial-patterns-data/uba_graph_preproc/y.csv") ###only float 대표###
    reg_df = pd.read_csv("C:/Users/sjjun/OneDrive/Desktop/spatial-patterns-data/uba_graph_preproc/reg.csv") ###GMT?###

    # load imputations of the models
    lin_imp = torch.load("C:/Users/sjjun/OneDrive/Desktop/spatial-patterns-data/modles/nearestneighborhybrid_predictions_useval.pt")
    rf_cs_imp = torch.load("C:/Users/sjjun/OneDrive/Desktop/spatial-patterns-data/modesl/RandomForest_cs_predictions_useval.pyt")
    
    ######데이터 보간 종료######

    # write imputations to gaps
    imp = y_true.copy()
    imp[missing_o3_mask] = rf_cs_imp[missing_o3_mask]
    gap_df_filtered = gap_df[(gap_df.type=='missing_o3_mask') &
                             (gap_df.len<6)]
    for gap_idx, gap_row in gap_df_filtered.iterrows():
        for idx in range(gap_row.start_idx, gap_row.start_idx+gap_row.len):
            if missing_o3_mask[idx]:
                imp[idx] = lin_imp[idx]

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
    o3_save_path = settings.resources_dir + 'imputed_dataset/o3.csv'
    info_save_path = settings.resources_dir + 'imputed_dataset/info.csv'
    imp_o3_df.to_csv(o3_save_path)
    print(f'written to {o3_save_path}')
    imp_info_df.to_csv(info_save_path)
    print(f'written to {info_save_path}')


if __name__ == '__main__':
    """
    Start or test routines.
    """
    create_final_imputation_ = True

    if create_final_imputation_:
        create_final_imputation()

