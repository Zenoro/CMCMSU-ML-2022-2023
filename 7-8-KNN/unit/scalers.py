import numpy as np


class MinMaxScaler:
    maximum = np.array([])
    minimum = np.array([])

    def fit(self, data):
        """Store calculated statistics

        Parameters:
        data (np.array): train set, size (num_obj, num_features)
        """
        self.maximum = np.max(data, axis=0)
        self.minimum = np.min(data, axis=0)

    def transform(self, data):
        """
        Parameters:
        data (np.array): train set, size (num_obj, num_features)

        Return:
        np.array: scaled data, size (num_obj, num_features)
        """
        data = data.T
        for i in range(len(data)):
            if self.maximum[i] != self.minimum[i]:
                data[i] = (data[i] - self.minimum[i]) / (self.maximum[i] - self.minimum[i])
            else:
                data[i] = 0
        return data.T


class StandardScaler:
    EM = np.array([])
    sigm = np.array([])

    def fit(self, data):
        """Store calculated statistics

        Parameters:
        data (np.array): train set, size (num_obj, num_features)
        """
        self.EM = np.mean(data, axis=0)
        self.sigm = np.std(data, axis=0)

    def transform(self, data):
        """
        Parameters:
        data (np.array): train set, size (num_obj, num_features)

        Return:
        np.array: scaled data, size (num_obj, num_features)
        """
        data = data.T
        for i in range(len(data)):
            if self.sigm[i] != 0:
                data[i] = (data[i] - self.EM[i]) / self.sigm[i]
            else:
                data[i] = 0
        return data.T
