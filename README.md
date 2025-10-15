# Task 2. Named entity recognition + image classification

This project implements a two-stage **Machine Learning pipeline** combining **Natural Language Processing (NLP)** and **Computer Vision (CV)** tasks.

The images were attached in a not very convenient zip file format due to their excessively large size. So for your convenience, I'm adding a link to the disk with the data - [Task2](https://drive.google.com/drive/folders/1E82LfOoWS8N13dv9tsFCNteqVp8qYIBJ?usp=sharing), containing all the files for this task.

## Setup
1. Clone the repo
    git clone https://github.com/Liudmyla03/DS_test_internship/tree/task_2.git
    cd task_1

2. Install dependencies
    pip install -r requirements.txt

3. Run the demo notebook
    jupyter notebook demo_notebook.ipynb

## Project Overview

The goal is to:
1. **Understand** what animal is mentioned in the user’s text using a **Named Entity Recognition (NER)** model.
2. **Classify** the animal in an input image using an **Image Classification CNN**.
3. **Compare** the two results and output a **boolean value** — `True` if the text matches the image, otherwise `False`.

### Example:
    Input text: "There is a cow in the picture."
    Input image: image_cow
    Output: True 

## Project Structure
task_2/
│
├── demo_notebook.ipynb # Dataset exploration and visualization
│
├── ner/
│ ├── train_ner.py # NER model training (transformer-based)
│ ├── infere_ner.py # NER inference script
│
├── image_classification/
│ ├── train_image.py # CNN model training
│ ├── infer_image.py # Model inference
│
├── pipeline/
│ └── pipeline.py # Unified text + image pipeline script
│
├── requirements.txt
├── README.md
└── data/
  ├── train/ # training images (10 classes)
  └── test/ # test images (10 classes)

## Dataset

**Source:** [Animals-10 dataset on Kaggle](https://www.kaggle.com/datasets/alessiocorrado99/animals10)

### Classes:
    dog, horse, elephant, butterfly, chicken, cat, cow, sheep, squirrel, spider
    Images are in .jpeg format and are divided into `train/` and `test/` folders.

## Tech stack
- Python 3 – main language
- NumPy, Pandas – data processing
- PyTorch, Torchvision – image classification (ResNet18)
- TensorFlow / Keras – optional CNN framework
- scikit-learn – preprocessing & metrics
- Transformers, Datasets, Evaluate, seqeval – NER model and evaluation
- Matplotlib, Seaborn – visualization
- tqdm, argparse, PyYAML, pathlib, json – utilities