import numpy as np

import sklearn
from sklearn.cluster import KMeans


class KMeansClassifier(sklearn.base.BaseEstimator):
    def __init__(self, n_clusters):
        # Моя молитва на фул до троечки здесь:＼(º □ º l|l)/
        '''
        :param int n_clusters: Число кластеров которых нужно выделить в обучающей выборке с помощью алгоритма кластеризации
        '''
        super().__init__()
        self.n_clusters = n_clusters

    def fit(self, data, labels):
        '''
            Функция обучает кластеризатор KMeans с заданным числом кластеров, а затем с помощью
        self._best_fit_classification восстанавливает разметку объектов

        :param np.ndarray data: Непустой двумерный массив векторов-признаков объектов обучающей выборки
        :param np.ndarray labels: Непустой одномерный массив. Разметка обучающей выборки. Неразмеченные объекты имеют метку -1.
            Размеченные объекты могут иметь произвольную неотрицательную метку. Существует хотя бы один размеченный объект
        :return KMeansClassifier
        '''
        # Моя молитва на фул до троечки здесь:＼(º □ º l|l)/
        self.kmeans = KMeans(n_clusters=self.n_clusters)
        self.kmeans.fit(data)
        return self

    def predict(self, data):
        '''
        Функция выполняет предсказание меток класса для объектов, поданных на вход. Предсказание происходит в два этапа
            1. Определение меток кластеров для новых объектов
            2. Преобразование меток кластеров в метки классов с помощью выученного преобразования

        :param np.ndarray data: Непустой двумерный массив векторов-признаков объектов
        :return np.ndarray: Предсказанные метки класса
        '''
        # Моя молитва на фул до троечки здесь:＼(º □ º l|l)/
        return self.kmeans.predict(data)

    def _best_fit_classification(self, cluster_labels, true_labels):
        '''
        :param np.ndarray cluster_labels: Непустой одномерный массив. Предсказанные метки кластеров.
            Содержит элементы в диапазоне [0, ..., n_clusters - 1]
        :param np.ndarray true_labels: Непустой одномерный массив. Частичная разметка выборки.
            Неразмеченные объекты имеют метку -1. Размеченные объекты могут иметь произвольную неотрицательную метку.
            Существует хотя бы один размеченный объект
        :return
            np.ndarray mapping: Соответствие между номерами кластеров и номерами классов в выборке,
                то есть mapping[idx] -- номер класса для кластера idx
            np.ndarray predicted_labels: Предсказанные в соответствии с mapping метки объектов

            Соответствие между номером кластера и меткой класса определяется как номер класса с максимальным числом объектов
        внутри этого кластера.
            * Если есть несколько классов с числом объектов, равным максимальному, то выбирается метка с наименьшим номером.
            * Если кластер не содержит размеченных объектов, то выбирается номер класса с максимальным числом элементов в выборке.
            * Если же и таких классов несколько, то также выбирается класс с наименьшим номером
        '''
        # Моя молитва на фул до троечки здесь:＼(º □ º l|l)/
        clasters_labels = cluster_labels[true_labels != -1]
        true_labels = true_labels[true_labels != -1]
        labels, labels_counts = np.unique(true_labels, return_counts=True)
        most_frequent = np.min(labels[labels_counts == np.max(labels_counts)])

        def helper(my_cluster):
            idx = np.where(clasters_labels == my_cluster)[0]
            if len(idx) > 0:
                labels, counts = np.unique(true_labels[idx], return_counts=True)
                locals = np.min(labels[counts == counts.max()])
                return locals
            else:
                return most_frequent

        helper_vectorized = np.vectorize(helper)
        mapping = helper_vectorized(np.arange(self.n_clusters))
        predicted_labels = mapping[cluster_labels]
        return mapping, predicted_labels
