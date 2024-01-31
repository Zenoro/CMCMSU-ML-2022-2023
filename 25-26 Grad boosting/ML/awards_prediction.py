from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd

import numpy as np  
from lightgbm import LGBMRegressor

from catboost import CatBoostRegressor
"""
 Внимание!
 В проверяющей системе имеется проблема с catboost.
 При использовании этой библиотеки, в скрипте с решением необходимо инициализировать метод с использованием `train_dir` как показано тут:
 CatBoostRegressor(train_dir='/tmp/catboost_info')
"""


def train_model_and_predict(train_file: str, test_file: str) -> np.ndarray:
    """
    This function reads dataset stored in the folder, trains predictor and returns predictions.
    :param train_file: the path to the training dataset
    :param test_file: the path to the testing dataset
    :return: predictions for the test file in the order of the file lines (ndarray of shape (n_samples,))
    """

    df_train = pd.read_json(train_file, lines=True)
    df_test = pd.read_json(test_file, lines=True)

    # remove categorical variables

    for column in ['genres', 'directors', 'filming_locations', 'keywords']:
        del df_train[column]
        del df_test[column]

    for i in range(3):
        del df_train[f"actor_{i}_gender"]
        del df_test[f"actor_{i}_gender"]

    y_train = df_train["awards"]
    del df_train["awards"]
    """
    2143
    """
    #regressor = LGBMRegressor(n_estimators=3000, learning_rate=0.01, max_depth=5, num_leaves=11)
    """
    2172
    """
    #regressor = LGBMRegressor(learning_rate=0.01, n_estimators=2000, max_depth=5, num_leaves=11, min_child_samples=8, reg_alpha=0.01, reg_lambda=0.1)
    """
    2128
    """
    regressor = CatBoostRegressor(iterations=4000, learning_rate=0.008, border_count=26, max_depth=7, eval_metric='MAE', l2_leaf_reg=2, silent=True)
    """govno"""
    #regressor = CatBoostRegressor(loss_function='MAE', iterations=2000, learning_rate=0.05, l2_leaf_reg=2,random_seed=42)
    #regressor = GradientBoostingRegressor(n_estimators=500)
    regressor.fit(df_train.to_numpy(), y_train.to_numpy())
    return regressor.predict(df_test.to_numpy())
