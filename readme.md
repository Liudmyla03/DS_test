# Models Overview

This folder contains three MNIST classification models that all implement  
the common interface `MnistClassifierInterface` (`train` and `predict` methods).

## 1️. Random Forest — `rf_model.py`
- **Library:** scikit-learn  
- **Description:** Baseline non-neural model using ensemble decision trees.  
- **Expected Accuracy:** ~95%.

## 2️. Feed-Forward Neural Network — `nn_model.py`
- **Library:** TensorFlow / Keras  
- **Description:** Fully connected network with several dense layers.  
- **Expected Accuracy:** ~92%.

## 3️. Convolutional Neural Network — `cnn_model.py`
- **Library:** TensorFlow / Keras  
- **Description:** CNN architecture with convolution, pooling, and dense layers.  
- **Expected Accuracy:** ~70%.