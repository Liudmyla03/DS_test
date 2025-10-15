
"""
Contains:
- MnistClassifierInterface (abstract base class)
- MnistClassifier (wrapper that chooses implementation by name: 'rf', 'nn', 'cnn')
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional


class MnistClassifierInterface(ABC):
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None, **kwargs):
        """
        Train the model
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for X
        """
        raise NotImplementedError


class MnistClassifier:
    """
    Wrapper class. Choose algorithm by name:
      - 'rf'  : Random Forest
      - 'nn'  : Feed-Forward Neural Network
      - 'cnn' : Convolutional Neural Network
    """

    def __init__(self, algorithm: str, **kwargs):
        algorithm = algorithm.lower()
        if algorithm == 'rf':
            from models.rf_model import RandomForestMnist
            self.model: MnistClassifierInterface = RandomForestMnist(**kwargs)
        elif algorithm == 'nn':
            from models.nn_model import FeedForwardMnist
            self.model: MnistClassifierInterface = FeedForwardMnist(**kwargs)
        elif algorithm == 'cnn':
            from models.cnn_model import CnnMnist
            self.model: MnistClassifierInterface = CnnMnist(**kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}. Use 'rf', 'nn' or 'cnn'.")

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        return self.model.train(X_train, y_train, X_val=X_val, y_val=y_val, **kwargs)

    def predict(self, X):
        return self.model.predict(X)
