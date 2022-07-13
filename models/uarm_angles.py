import joblib
from models.regressor import Regressor
import config as cfg
import numpy as np
import pandas as pd


class UarmAngles(Regressor):
    def __init__(self):
        if cfg.VIDEO_TYPE == 'test':
            df_angles_path = 'data/20210103111902_TESTING.csv'
            if cfg.CAMERA == 'left':
                model_path = 'models/interpolator_left_test.joblib'
            else:
                model_path = 'models/interpolator_right_test.joblib'

        else:
            df_angles_path = 'data/20210103110338.csv'

            if cfg.CAMERA == 'left':
                model_path = 'models/interpolator_left_train.joblib'
            else:
                model_path = 'models/interpolator_right_train.joblib'
        df_angles = pd.read_csv(df_angles_path, header=0)
        self.t0 = df_angles['timestamp'][0]
        self.interp_alpha_0, self.interp_alpha_1, self.interp_alpha_2 = joblib.load(model_path)

    def predict(self, x):
        x_normalized = x - self.t0
        return np.array([self.interp_alpha_0(x_normalized).item(), self.interp_alpha_1(x_normalized).item(),
                         self.interp_alpha_2(x_normalized).item()])
