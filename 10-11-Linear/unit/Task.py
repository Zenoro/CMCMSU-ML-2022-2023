import numpy as np


def correctit(xx):
    i = 0
    mask = []
    elems = list(set(xx))
    if any((type(i) == str for i in xx)):
        for i in xx:
            if all((i in mask for i in elems)):
                break
            elif i not in mask:
                mask.append(i)
    for i in range(len(mask)):
        xx[xx == mask[i]] = i
    return xx


class Preprocessor:

    def __init__(self):
        pass

    def fit(self, X, Y=None):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X, Y=None):
        pass


class MyOneHotEncoder(Preprocessor):

    def __init__(self, dtype=np.float64):
        super(Preprocessor).__init__()
        self.dtype = dtype

    def fit(self, X, Y=None):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: unused
        """
        self.names = [col for col in X.columns]
        self.str_exception = [i for i in self.names if X.dtypes[i] != int]

    def transform(self, X):
        """
        param    X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        returns: transformed objects, numpy-array, shape [n_objects, |f1| + |f2| + ...]
        """

        def fun(mat):
            b = np.zeros((mat.size, mat.max() + 1))
            for j in np.arange(mat.size):
                b[j, mat[j]] = 1
            return b

        work = X.to_numpy().T
        if self.str_exception:
            for i in self.str_exception:
                work[X.columns.get_loc(i)] = correctit(X[i].to_numpy())

        res = fun(work[0])
        for i in work[1:]:
            res = np.append(res, fun(i), axis=1)
        return res

    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


class SimpleCounterEncoder:

    def __init__(self, dtype=np.float64):
        self.dtype = dtype

    def fit(self, X, Y):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        """
        self.names = []
        work = X.to_numpy().T
        target = Y.to_numpy().T
        for i in range(work.shape[0]):
            d = dict()
            for k in np.unique(work[i]):
                temp = np.where(work[i] == k)
                x1 = np.sum(target[temp]) / len(temp[0])
                x2 = np.sum([work[i] == k]) / len(work[i])
                d[k] = [x1, x2]
            self.names.append(d)

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3]
        """
        work = X.to_numpy()
        finres = np.zeros((X.shape[0], 3*len(self.names)))
        for i in range(work.shape[0]):
            res = []
            for j in range(work.shape[1]):
                buf = self.names[j].get(work[i][j])
                temp = (buf[0] + a) / (buf[1] + b)
                res += buf + [temp]
            finres[i] = res
        return finres

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


def group_k_fold(size, n_splits=3, seed=1):
    idx = np.arange(size)
    np.random.seed(seed)
    idx = np.random.permutation(idx)
    n_ = size // n_splits
    for i in range(n_splits - 1):
        yield idx[i * n_: (i + 1) * n_], np.hstack((idx[:i * n_], idx[(i + 1) * n_:]))
    yield idx[(n_splits - 1) * n_:], idx[:(n_splits - 1) * n_]


class FoldCounters:

    def __init__(self, n_folds=3, dtype=np.float64):
        self.dtype = dtype
        self.n_folds = n_folds

    def fit(self, X, Y, seed=1):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        param seed: random seed, int
        """
        self.foldslice = list(group_k_fold(X.shape[0], self.n_folds, seed))
        self.names = []
        for targint, trainind in self.foldslice:
            work = X.to_numpy()[trainind].T
            target = Y.to_numpy()[trainind].T
            buf = []
            for i in range(X.shape[1]):
                d = dict()
                for k in np.unique(work[i]):
                    temp = np.where(work[i] == k)
                    x1 = np.sum(target[temp]) / len(temp[0])
                    x2 = np.sum([work[i] == k]) / len(work[i])
                    d[k] = [x1, x2]
                buf.append(d)
            self.names.append(buf)

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3]
        """
        work = X.to_numpy()
        finres = np.zeros((X.shape[0], 3*X.shape[1]))
        for arr in self.foldslice:
            dicbuf = self.names[0]
            for elem in arr[0]:
                buf = work[elem]
                res = []
                for k in range(len(buf)):
                    resbuf = dicbuf[k].get(buf[k])
                    temp = (resbuf[0] + a) / (resbuf[1] + b)
                    res += resbuf + [temp]
                finres[elem] = res
            self.names.pop(0)
        return finres

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)


def weights(x, y):
    """
    param x: training set of one feature, numpy-array, shape [n_objects,]
    param y: target for training objects, numpy-array, shape [n_objects,]
    returns: optimal weights, numpy-array, shape [|x unique values|,]
    """
    sample_data = sorted(list(set(x)))
    data_count = dict((c, 0) for c in sample_data)
    data_dict = dict()
    weights = np.array([])
    for i in range(len(x)):
        if (y[i]) == 1:
            data_count[x[i]] += 1
    for k in sorted(x):
        data_dict[k] = data_dict.get(k, 0)+1
    for r in sample_data:
        ans = 0
        ans = data_count[r] / data_dict[r]
        weights = np.append(weights, ans)
    return weights
