"""
Correct and smooth algorithm for missing data imputation of the
UBA dataset. These are the experiments we are publishing.

Run apply_correct_and_smooth(save=False) to print the evaluations as
published.
"""

# general
import pdb
import pickle as pkl

# data science
import numpy as np
import pandas as pd

# sklearn
from sklearn.metrics import mean_squared_error, r2_score

# pytorch
import torch

# plotting
import matplotlib.pyplot as plt

# own package
import settings
from preprocessing_uba import UBAGraph
from models_correct_and_smooth import CorrectAndSmooth
from models_simple import SpatiotemporalMean
from models_simple import SpatialMean
from models_simple import NearestNeighborHybrid
from models_simple import RandomForest
from postprocessing_evaluation_tools import index_of_agreement
from postprocessing_evaluation_tools import GapLengthEvaluation



class CSWorkflow:
    """
    A class for all correct and smooth workflows
    """
    def __init__(self, simple_model, use_val=False):
        """
        Initialize the workflow with a simple model and
        indicate whether correct and smooth should be performed.
        """
        self.simple_model_class = simple_model
        self.use_val = use_val

    def get_dataset(self):
        """
        Read in the UBAGraph dataset
        """
        ug = UBAGraph()
        ug.get_dataset()
        dataset = ug
        self.data = dataset[0]

    def get_simple_model(self):
        """
        Read predictions of simple model
        """
        self.sm = self.simple_model_class(use_val=self.use_val)
        print(f'\nRead predictions for {self.sm} from {self.sm.prediction_path}')
        self.sm.read_predictions()
        self.y_hat_simple_model = self.sm.y_hat

    def correct_and_smooth(self, correction_alpha, num_correction_layers,
                           smoothing_alpha, num_smoothing_layers, scale):
        """
        Correct and smooth post processing of the
        simple model
        """
        #print('\nCorrect and smooth')
        print(f'  Correction alpha {correction_alpha:.1f}')
        print(f'  Number of correction layers {num_correction_layers}')
        print(f'  Smoothing alpha {smoothing_alpha:.1f}')
        print(f'  Number of smoothing layers {num_smoothing_layers}')
        print(f'  Scale {scale:.2f}')
        if self.use_val:
            mask = self.data.train_mask | self.data.val_mask
        else:
            mask = self.data.train_mask
        cs = CorrectAndSmooth(num_correction_layers=num_correction_layers,
                              correction_alpha=correction_alpha,
                              num_smoothing_layers=num_smoothing_layers,
                              smoothing_alpha=smoothing_alpha,
                              autoscale=False,
                              scale=scale)
        self.y_hat_correct = cs.correct(y_soft=self.y_hat_simple_model,
                                        y_true=self.data.y[mask],
                                        mask=mask,
                                        edge_index=self.data.edge_index,
                                        edge_weight=self.data.edge_weight)
        self.y_hat_smooth = cs.smooth(y_soft=self.y_hat_correct,
                                      y_true=self.data.y[mask],
                                      mask=mask,
                                      edge_index=self.data.edge_index,
                                      edge_weight=self.data.edge_weight)
        self.y_hat_smooth[mask] = self.data.y[mask]

    def evaluate(self):
        """
        R2, RMSE of simple model and correct and smooth
        """
        print('\nEvaluation')
        for model_name, y_hat in [('Simple model', self.y_hat_simple_model),
                                  #('Correct', self.y_hat_correct),
                                  ('Smooth', self.y_hat_smooth)]:
            print(f'  {model_name}')
            if self.use_val:
                mask_specs = [#('Training+Validation',self.data.train_mask|self.data.val_mask),
                              ('Test', self.data.test_mask)]
            else:
                mask_specs = [('Training', self.data.train_mask),
                              ('Validation', self.data.val_mask)]

            for mask_name, mask in mask_specs:
                print(f'    {mask_name}')
                y = self.data.y[mask].numpy()
                y_hat_ = y_hat[mask].numpy()
                r2 = r2_score(y, y_hat_)
                rmse = (mean_squared_error(y, y_hat_))**.5
                d = index_of_agreement(y, y_hat_)
                print(f'{r2:.4f} {rmse:.4f} {d:.4f}')
                return_r2 = r2
                return_rmse = rmse
                return_d = d
        if self.use_val:
            evaluation_set = 'test'
        else:
            evaluation_set = 'val'
        #print(f'Bin evaluation on {evaluation_set} set')
        gle = GapLengthEvaluation()
        #print(gle.evaluate_bins_separately(evaluation_set+'_mask', self.data.y, y_hat))
        #print()
        return return_r2, return_rmse, return_d


def tune_correct_and_smooth():
    """
    Tune the two parameters for correct and smooth
    """
    print(f"\n{'*'*29}\n** TUNE CORRECT AND SMOOTH **\n{'*'*29}\n")
    simple_models = [SpatiotemporalMean, SpatialMean,
                     NearestNeighborHybrid, RandomForest]
    correction_alphas = np.arange(0., 1.1, .2)
    num_correction_layers = [5, 10, 15, 20]
    smoothing_alphas = np.arange(0., 1.1, .2)
    num_smoothing_layers = [5, 10, 15, 20]
    scales = [.25, .5, .75, 1, 1.25, 1.5, 1.75, 2., 2.25, 2.5 ]

    specs = [(sm, ca, ncl, sa, nsl, sc) for sm in simple_models
                                        for ca in correction_alphas
                                        for ncl in num_correction_layers
                                        for sa in smoothing_alphas
                                        for nsl in num_smoothing_layers
                                        for sc in scales]

    columns = ['simple_model', 'correction_alpha',
               'num_correction_layers', 'smoothing_alpha',
               'num_smoothing_layers', 'scale']
    tuning_df = pd.DataFrame(columns=columns, data=specs)
    tuning_df['r2'], tuning_df['rmse'], tuning_df['d'] = np.nan, np.nan, np.nan

    for count, (simple_model, correction_alpha, num_correction_layers,
     smoothing_alpha, num_smoothing_layers, scale) in enumerate(specs):
        csw = CSWorkflow(simple_model=simple_model)
        csw.get_dataset()
        csw.get_simple_model()
        csw.correct_and_smooth(correction_alpha=correction_alpha,
                               num_correction_layers=num_correction_layers,
                               smoothing_alpha=smoothing_alpha,
                               num_smoothing_layers=num_smoothing_layers,
                               scale=scale)
        val_r2, val_rmse, val_d = csw.evaluate()
        tuning_df.loc[count, ['r2', 'rmse', 'd']] = round(val_r2, 3), \
                                    round(val_rmse, 3), round(val_d, 3)
        if count % 10 == 0: tuning_df.to_csv(settings.output_dir+'tuning.csv')

    tuning_df.to_csv(settings.output_dir+'tuning.csv')
    print(f'tuning results written to {settings.output_dir}tuning.csv')


def apply_correct_and_smooth(save=False):
    """
    Finally, apply correct using training and validation set!
    If save is set to true, the saved predictions are overwritten.
    """
    print(f"\n{'*'*30}\n** APPLY CORRECT AND SMOOTH **\n{'*'*30}\n")
    rf_hyperparameter = settings.rf_hyperparameter
    specs = [
             #(SpatiotemporalMean, 1.0, 20, 1.0, 20, 1.0),  # final
             #(SpatialMean, 1.0, 20, 1.0, 10, 0.5),  # final
             #(NearestNeighborHybrid, 0.0, 5, 1.0, 10, 0.25),  # final
             (RandomForest, *rf_hyperparameter),  # final
             ] ###Parameter: Correction alpha, Number of correction layers, Smoothing alpha, Number of smoothing layers, Scale

    for simple_model, correction_alpha, num_correction_layers, \
        smoothing_alpha, num_smoothing_layers, scale in specs:
        csw = CSWorkflow(simple_model=simple_model, use_val=True)
        csw.get_dataset()
        csw.get_simple_model()
        csw.correct_and_smooth(correction_alpha=correction_alpha,
                               num_correction_layers=num_correction_layers,
                               smoothing_alpha=smoothing_alpha,
                               num_smoothing_layers=num_smoothing_layers,
                               scale=scale)


        dir_ = settings.resources_dir + 'models/'
        file_ = f'{simple_model()}_cs_predictions_useval.pyt'
        torch.save(csw.y_hat_smooth, dir_+file_)
        print(f'written to {dir_+file_}\n\n')
        csw.evaluate()
    
    


if __name__ == '__main__':
    """
    Start a workflow
    """
    #tune_correct_and_smooth()
    apply_correct_and_smooth()




