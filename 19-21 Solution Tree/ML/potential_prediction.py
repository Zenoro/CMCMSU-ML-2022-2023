import os

from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV

import numpy as np


class PotentialTransformer:
    """
    A potential transformer.

    This class is used to convert the potential's 2d matrix to 1d vector of features.
    """

    def fit(self, x, y):
        """
        Build the transformer on the training set.
        :param x: list of potential's 2d matrices
        :param y: target values (can be ignored)
        :return: trained transformer
        """
        return self

    def fit_transform(self, x, y):
        """
        Build the transformer on the training set and return the transformed dataset (1d vectors).
        :param x: list of potential's 2d matrices
        :param y: target values (can be ignored)
        :return: transformed potentials (list of 1d vectors)
        """
        return self.transform(x)

    def transform(self, x):
        """
        Transform the list of potential's 2d matrices with the trained transformer.
        :param x: list of potential's 2d matrices
        :return: transformed potentials (list of 1d vectors)
        """
        return x.reshape((x.shape[0], -1))


def load_dataset(data_dir):
    files, X, Y = [], [], []
    for file in os.listdir(data_dir):
        potential = np.load(os.path.join(data_dir, file))
        files.append(file)
        X.append(potential["data"])
        Y.append(potential["target"])
    return files, np.array(X), np.array(Y)


def train_model_and_predict(train_dir, test_dir):
    _, X_train, Y_train = load_dataset(train_dir)
    test_files, X_test, _ = load_dataset(test_dir)
    pipe = Pipeline([('vectorizer', PotentialTransformer()), ('reg', ExtraTreesRegressor())])
    myparams = {"reg__n_estimators": (10, 50),
                "reg__max_features": [*(np.arange(0.1, 1.1, 0.1))],
                "reg__random_state": list(range(4)),
                "reg__warm_start": (False, True)
                }
    regressor = GridSearchCV(pipe, myparams, cv=3, n_jobs=2, verbose=1000)
    regressor.fit(X_train, Y_train)
    predictions = regressor.predict(X_test)
    print("BEST:\t", regressor.best_estimator_)
    return {file: value for file, value in zip(test_files, predictions)}
