from collections import defaultdict
import numpy as np
from sklearn.metrics import r2_score
from sklearn import neighbors
from sklearn.preprocessing import MinMaxScaler


import numpy as np
from collections import defaultdict


def kfold_split(num_objects, num_folds):
    """Split [0, 1, ..., num_objects - 1] into equal num_folds folds (last fold can be longer) and returns num_folds train-val
       pairs of indexes.

    Parameters:
    num_objects (int): number of objects in train set
    num_folds (int): number of folds for cross-validation split

    Returns:
    list((tuple(np.array, np.array))):
    list of length num_folds, where i-th element of list contains tuple of 2 numpy arrays,
    the 1st numpy array contains all indexes without i-th fold while the 2nd one contains i-th fold
    """

    lenfold = num_objects // num_folds
    res = []
    a = np.arange(num_objects)
    for i in range(num_folds):
        y = a[0:(i * lenfold)]
        z = a[((i + 1) * lenfold):num_objects]
        y = np.append(y, z)
        res.append((y, a[(i * lenfold):(i + 1) * lenfold]))
    u = res.pop(-1)
    x, y = u[0], u[1]
    for i in range(num_objects % num_folds):
        last = x[-1]
        x = x[:-1]
        y = np.append(y, last)
    res.append((x, np.sort(y)))
    return res


def knn_cv_score(X, y, parameters, score_function, folds, knn_class):
    """Takes train data, counts cross-validation score over grid of parameters (all possible parameters combinations)

    Parameters:
    X (2d np.array): train set
    y (1d np.array): train labels
    parameters (dict): dict with keys from {n_neighbors, metrics, weights, normalizers}, values of type list,
                       parameters['normalizers'] contains tuples (normalizer, normalizer_name), see parameters
                       example in your jupyter notebook
    score_function (callable): function with input (y_true, y_predict) which outputs score metric
    folds (list): output of kfold_split
    knn_class (obj): class of knn model to fit

    Returns:
    dict: key - tuple of (normalizer_name, n_neighbors, metric, weight), value - mean score over all folds
    """

    keyres = []
    ressum = []
    output = {}
    for i in range(len(parameters['normalizers'])):
        X_transform = X
        if parameters["normalizers"][i][0] is not None:
            scalemethod = parameters["normalizers"][i][0]
            scalemethod.fit(X)
            # X_transform = scalemethod.fit_transform(X)
            X_transform = scalemethod.transform(X)
        keyres.append(parameters['normalizers'][i][1])
        for j in range(len(parameters['n_neighbors'])):
            keyres.append(parameters['n_neighbors'][j])
            for k in range(len(parameters['metrics'])):
                keyres.append(parameters['metrics'][k])
                for l in range(len(parameters['weights'])):
                    keyres.append(parameters['weights'][l])
                    for m in folds:
                        X_data = X_transform[m[0],:]
                        y_data = y[m[0]]
                        y_test = y[m[1]]
                        X_test = X_transform[m[1],:]
                        somclass = knn_class(n_neighbors=parameters['n_neighbors'][j], weights=parameters['weights'][l], metric=parameters['metrics'][k])
                        somclass.fit(X_data, y_data)
                        y_pred = somclass.predict(X_test)
                        ressum.append(score_function(y_test, y_pred))
                    output[tuple(keyres)] = sum(ressum)/len(ressum)
                    keyres = keyres[:-1]
                    ressum = []
                keyres = keyres[:-1]
            keyres = keyres[:-1]
        keyres = []
    return output


X_train = np.array([[ 0.62069296, -0.07097426,  0.65172896, -1.14620331],
                    [ 2.03347616,  0.32524614, -0.71941433, -0.30789854],
                    [ 0.17100377,  1.63120292,  1.34284446, -2.16397238],
                    [-1.65370417,  0.62499229, -0.50217293,  2.07813591],
                    [ 0.84667916,  0.25458428,  0.14720704, -0.18668345],
                    [ 0.43833344, -1.40348048, -1.37944118,  0.19192659],
                    [ 0.97229574, -0.54606276, -0.09855294,  1.28961291],
                    [ 0.25355626, -1.72816511,  0.084554  , -2.14256875],
                    [ 0.36103462, -1.28930935,  1.34586369, -0.57300728],
                    [-1.42711933, -0.11832827, -0.58038295, -1.56806583]])
y_train = np.sum(np.abs(X_train), axis=1)
scaler = MinMaxScaler()
parameters = {
    'n_neighbors': [1, 2, 4],
    'metrics': ['euclidean', 'cosine'],
    'weights': ['uniform', 'distance'],
    'normalizers': [(scaler, 'MinMaxScaler')]
}
folds = kfold_split(10, 3)
out = knn_cv_score(X_train, y_train, parameters, r2_score, folds, neighbors.KNeighborsRegressor)
print(out)









-3.886914104013526
-3.886914104013526
-3.339427669385479
-3.3394276693854805
# -4.190559808371016
# -4.086642592839364
-3.0373126577073877
-2.6002141232270652
# -0.8605748631378208
-1.1320958884621681
-0.33412505845764784
-0.6946087109718427

-3.886914104013526
-3.886914104013526
-3.339427669385479
-3.339427669385479
# -2.821522070818421
# -2.909515284414977
-3.0373126577073877
-2.7197802024893374
# -1.229118435031323
-1.4848798788742938
-0.3586698577110674
-0.7914850319051477


