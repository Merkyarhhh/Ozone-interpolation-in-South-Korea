"""
This file contains paths to all folders
and data files that we use in our project.
"""


# basic packages
import os
import pathlib
import pandas as pd
import pdb

# find the source of our project, only to know the directory
import source

# find the root directory
SOURCEFILE = os.path.abspath("C:/Users/ATMOS/Desktop/source/settings.py")
SOURCEDIR_pos = pathlib.Path(SOURCEFILE).parent
ROOTDIR = str(SOURCEDIR_pos.parent)

# data resources directory
resources_dir = 'C:/Users/ATMOS/Desktop/data/data_jeolla/'

# for plots, evaluation etc...
output_dir = resources_dir + 'output/'

# random seed
random_seed = 1

# global model/data settings
datetime_start = '2021-01-01 00:00:00'  # start of dataset in UTC
datetime_end = '2021-12-31 23:59:59'  # end of dataset in UTC
time_offset = pd.Timedelta('0 days 00:00:00')  # UTC to local, but already local time data

radius = 4.  # radius to be connected by edge
time_window = pd.Timedelta('0 days 01:00:00')  # time diff for edge
# 1-2h, 2-3h, 3-6h, 6h-1d, 1d-1w, 1w-1y
bins = [1, 2, 3, 6, 24, 168, 8761]

#NNH hyperparameter
limit = 4

#Random forest feature selection
#features = ['hour_of_day', 'day_of_year', 'cams_no2', 'temperature', 'day_of_week', 'cams_pm10', 'cams_pm25']
features = ['hour_of_day',
            'day_of_week',
            'day_of_year',
            'cams_no2',
            'cams_co',
            'cams_so2',
            'cams_pm10',
            'cams_pm25',
            'temperature',
            'wind_speed',
            'relative_humidity',
            ]
features_not = [#'hour_of_day',
                #'day_of_week',
                #'day_of_year',
                #'cams_no2',
                #'cams_co',
                #'cams_so2',
                #'cams_pm10',
                #'cams_pm25',
                #'temperature',
                'wind_speed',
                #'relative_humidity',
                ]
print(features_not)
rf_hyperparameter = 1,20,0.8,5,0.75
# for debug
# pd.set_option('display.max_rows', 100)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)

if __name__ == '__main__':
    print('SOURCEDIR:', SOURCEDIR_pos)
    print('ROOTDIR:', ROOTDIR)
    print('output_dir:', output_dir)
    print('resources_dir:', resources_dir)
    print(features_not)

#exec(open("C:/Users/ATMOS/Desktop/source/retrieval_cams.py").read())
#exec(open("C:/Users/ATMOS/Desktop/source/preprocessing_uba.py").read())
exec(open("C:/Users/ATMOS/Desktop/source/models_simple.py").read())
exec(open("C:/Users/ATMOS/Desktop/source/experiments_cs_uba_graph.py").read())
#exec(open("C:/Users/ATMOS/Desktop/source/postprocessing_create_final_imputation.py").read())
#exec(open("C:/Users/ATMOS/Desktop/source/postprocessing_evaluation_tools.py").read())
#exec(open("C:/Users/ATMOS/Desktop/source/draw_scatter.py").read())
#exec(open("C:/Users/ATMOS/Desktop/source/visualizations_plots_for_paper.py").read())
