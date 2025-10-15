"""
Convolutional Neural Network (Keras) implementation for MNIST
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from mnist_interface import MnistClassifierInterface


class CnnMnist(MnistClassifierInterface):
    def __init__(self, epochs: int = 10, batch_size: int = 128):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

    def _build(self, input_shape, n_classes):
        """
        input_shape: e.g. (28,28,1)
        """
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(n_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model = model

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val=None, y_val=None, **kwargs):
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
