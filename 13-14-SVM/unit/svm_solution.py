from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


def train_svm_and_predict(train_features, train_target, test_features):
    """
    train_features: np.array, (num_elements_train x num_features) - train data description, the same features and the same order as in train data
    train_target: np.array, (num_elements_train) - train data target
    test_features: np.array, (num_elements_test x num_features) -- some test data, features are in the same order as train features

    return: np.array, (num_elements_test) - test data predicted target, 1d array
    """

    scaler = StandardScaler()
    scaler.fit(train_features)
    X_train_scaled = scaler.transform(train_features)
    X_test_scaled = scaler.transform(test_features)

    model = SVC(kernel='rbf', C=2.5, gamma='auto', class_weight='balanced', degree=5)
    model.fit(X_train_scaled, train_target)
    return model.predict(X_test_scaled)
