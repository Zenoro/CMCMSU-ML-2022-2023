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
        keyres.append(parameters['normalizers'][i][1])
        for j in range(len(parameters['n_neighbors'])):
            keyres.append(parameters['n_neighbors'][j])
            for k in range(len(parameters['metrics'])):
                keyres.append(parameters['metrics'][k])
                for n in range(len(parameters['weights'])):
                    keyres.append(parameters['weights'][n])
                    for m in folds:
                        if len(X.shape) > 1:
                            X_data = X[m[0], :]
                            X_test = X[m[1], :]
                        else:
                            X_data = X[m[0]]
                            X_test = X[m[1]]
                        y_data = y[m[0]]
                        y_test = y[m[1]]
                        if parameters["normalizers"][i][0] is not None:
                            scalemethod = parameters["normalizers"][i][0]
                            scalemethod.fit(X_data)
                            X_data = scalemethod.transform(X_data)
                            X_test = scalemethod.transform(X_test)
                        somclass = knn_class(n_neighbors=parameters['n_neighbors'][j], weights=parameters['weights'][n], metric=parameters['metrics'][k])
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
