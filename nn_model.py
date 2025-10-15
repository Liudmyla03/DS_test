"""
Feed-Forward Neural Network (Keras) implementation for MNIST
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from mnist_interface import MnistClassifierInterface


class FeedForwardMnist(MnistClassifierInterface):
    def __init__(self, hidden_units=(256, 128), epochs: int = 10, batch_size: int = 128):
        self.hidden_units = list(hidden_units)
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

    def _build(self, input_shape, n_classes):
        """
        Build simple feed-forward model
        """
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        for units in self.hidden_units:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(0.2))
        model.add(Dense(n_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model = model

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val=None, y_val=None, **kwargs):
        # determine number of classes and build model
        n_classes = int(np.max(y_train) + 1)
        self._build(input_shape=X_train.shape[1:], n_classes=n_classes)
        callbacks = [EarlyStopping(patience=3, restore_best_weights=True)]
        self.model.fit(X_train, y_train,
                       validation_data=(X_val, y_val) if X_val is not None else None,
                       epochs=self.epochs,
                       batch_size=self.batch_size,
                       callbacks=callbacks,
                       verbose=2)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = self.model.predict(X)
        return preds.argmax(axis=1)
