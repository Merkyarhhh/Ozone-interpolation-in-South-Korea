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
    ug = UBAGraph()
    gap_df = pd.read_csv(ug.gap_path, index_col=0)
    mask_df = pd.read_csv(ug.mask_path, index_col=0)
    missing_o3_mask = mask_df.missing_o3_mask.to_numpy().reshape(-1)
    y_true_df = pd.read_csv(ug.y_path, index_col=0)
    y_true = y_true_df.y.to_numpy().reshape(-1)
    reg_df = pd.read_csv(ug.reg_path, index_col=0)

    # load imputations of the models
    model_path = settings.resources_dir + 'models/'
    sm_file = 'spatialmean_predictions_useval.pt'
    nnh_file = 'nearestneighborhybrid_predictions_useval.pt'
    rf_file = 'randomforest_predictions_useval.pt'
    sm_cs_file = 'SpatialMean_cs_predictions_useval.pyt'
    nnh_cs_file = 'NearestNeighborHybrid_cs_predictions_useval.pyt'
    rf_cs_file = 'RandomForest_cs_predictions_useval.pyt'
    
    sm_imp = torch.load(model_path+sm_file).numpy().reshape(-1)
    nnh_imp = torch.load(model_path+nnh_file).numpy().reshape(-1)
    rf_imp = torch.load(model_path+rf_file).numpy().reshape(-1)
    sm_cs_imp = torch.load(model_path+sm_cs_file).numpy().reshape(-1)
    nnh_cs_imp = torch.load(model_path+nnh_cs_file).numpy().reshape(-1)
    rf_cs_imp = torch.load(model_path+rf_cs_file).numpy().reshape(-1)

    # write imputations to gaps
    imp = y_true.copy()
    imp[missing_o3_mask] = rf_cs_imp[missing_o3_mask]
    
    gap_df_filtered = gap_df[(gap_df.type=='missing_o3_mask') & (gap_df.len>=1) & (gap_df.len<2)]
    for gap_idx, gap_row in gap_df_filtered.iterrows():
        for idx in range(gap_row.start_idx, gap_row.start_idx+gap_row.len):
            if missing_o3_mask[idx]:
                imp[idx] = nnh_imp[idx]
    
    gap_df_filtered = gap_df[(gap_df.type=='missing_o3_mask') & (gap_df.len>=2) & (gap_df.len<3)]
    for gap_idx, gap_row in gap_df_filtered.iterrows():
        for idx in range(gap_row.start_idx, gap_row.start_idx+gap_row.len):
            if missing_o3_mask[idx]:
                imp[idx] = nnh_imp[idx]
                
    gap_df_filtered = gap_df[(gap_df.type=='missing_o3_mask') & (gap_df.len>=3) & (gap_df.len<6)]
    for gap_idx, gap_row in gap_df_filtered.iterrows():
        for idx in range(gap_row.start_idx, gap_row.start_idx+gap_row.len):
            if missing_o3_mask[idx]:
                imp[idx] = nnh_imp[idx]
    
    gap_df_filtered = gap_df[(gap_df.type=='missing_o3_mask') & (gap_df.len>=6) & (gap_df.len<24)]
    for gap_idx, gap_row in gap_df_filtered.iterrows():
        for idx in range(gap_row.start_idx, gap_row.start_idx+gap_row.len):
            if missing_o3_mask[idx]:
                imp[idx] = rf_cs_imp[idx]
    
    gap_df_filtered = gap_df[(gap_df.type=='missing_o3_mask') & (gap_df.len>=24) & (gap_df.len<168)]
    for gap_idx, gap_row in gap_df_filtered.iterrows():
        for idx in range(gap_row.start_idx, gap_row.start_idx+gap_row.len):
            if missing_o3_mask[idx]:
                imp[idx] = rf_cs_imp[idx]
            
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
    imp_o3_df.index.name = 'date'

    # prepare an info dataframe
    imp_info_df = pd.DataFrame(index=datetime_list,
                               columns=station_list,
                               data=missing_o3_mask.reshape(n_stations,
                                                            n_datetime).T)
    imp_info_df.index.name = 'date'

    # save data
    o3_save_path = settings.resources_dir + 'imputed_dataset/imputed_o3.csv'
    info_save_path = settings.resources_dir + 'imputed_dataset/imputed_info.csv'
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

