# Regularization_Deep_learning
Overview
This Jupyter Notebook demonstrates Regularization techniques in Deep Learning using the Iris dataset. Regularization methods help prevent overfitting and improve generalization in neural networks.

Objectives
Understand the need for regularization in deep learning.

Implement L1 (Lasso) and L2 (Ridge) regularization.

Apply Dropout to improve model robustness.

Evaluate model performance with and without regularization.

Dataset
The Iris dataset is used, containing 150 samples from three different species of iris flowers:

Features: Sepal Length, Sepal Width, Petal Length, Petal Width

Target: Species (Setosa, Versicolor, Virginica)

Implementation Steps
Import Required Libraries

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
Load and Preprocess Data

Convert categorical labels to numeric using Label Encoding.

Normalize features using StandardScaler.

Build Neural Network Model

Define a simple feedforward neural network using Keras.

Apply L1/L2 regularization to dense layers.

Use Dropout layers for additional regularization.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l1, l2

model = Sequential([
    Dense(64, activation='relu', kernel_regularizer=l2(0.01), input_shape=(4,)),
    Dropout(0.5),
    Dense(32, activation='relu', kernel_regularizer=l1(0.01)),
    Dropout(0.5),
    Dense(3, activation='softmax')
])
Compile and Train the Model

python:
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

Evaluate Model Performance
Plot training and validation accuracy/loss.
Compare models with and without regularization.

Results & Observations
Without regularization, the model may overfit (high training accuracy, low validation accuracy).
With L1/L2 regularization, the model generalizes better.
Dropout further improves model robustness.

Conclusion
Regularization techniques like L1, L2, and Dropout help control overfitting and improve the generalization ability of deep learning models.

Dependencies
Ensure you have the following installed:
pip install tensorflow numpy pandas scikit-learn matplotlib
