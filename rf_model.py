"""
Random Forest implementation for MNIST
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from mnist_interface import MnistClassifierInterface


class RandomForestMnist(MnistClassifierInterface):
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.clf = RandomForestClassifier(n_estimators=self.n_estimators,
                                          random_state=self.random_state)

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val=None, y_val=None, **kwargs):
        """
        Flatten inputs and fit RandomForest
        """
        X = X_train.reshape((X_train.shape[0], -1))
        self.clf.fit(X, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Flatten inputs and predict
        """
        Xf = X.reshape((X.shape[0], -1))
        return self.clf.predict(Xf)
