# Task 1. Image classification + OOP
## Description

This project implements three classifiers for the MNIST handwritten digit dataset using an object-oriented approach.
Each model (Random Forest, Feed-Forward NN, CNN) implements the same interface — MnistClassifierInterface — with the following methods:

    train(X_train, y_train)
    predict (X_test)

The MnistClassifier wrapper class selects a model by name (‘rf’, “nn”, ‘cnn’), providing unified input/output formats.

## Setup
1. Clone the repo
    git clone https://github.com/Liudmyla03/DS_test_internship/tree/task_1.git
    cd task_1

2. Install dependencies
    pip install -r requirements.txt

3. Run the demo notebook
    jupyter notebook notebook_demo.ipynb

## Structure
    task_1/
    ├─ mnist_interface.py   # Interface + model selector
    ├─ models/
    │  ├─ rf_model.py       # Random Forest
    │  ├─ nn_model.py       # Feed-Forward NN
    │  └─ cnn_model.py      # Convolutional NN
    ├─ utils.py             # Load and preprocess MNIST
    ├─ notebook_demo.ipynb  # Example of training and prediction
    └─ requirements.txt

## Result
| Model           | Accuracy (approx.) |
| --------------- | ------------------ |
| Random Forest   | ~95%               |
| Feed-Forward NN | ~90%               |
| CNN             | ~70%               |

## Tech stack
- Python 3
- NumPy
- Matplotlib
- scikit-learn

- TensorFlow / Keras
