# imports
import sys
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from tensorflow.keras import Sequential
from keras.models import Model
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split



def mlp_model(input_shape: tuple, multi: bool):
    """Multi Layer Perceptron model
    4 hidden layers, with decreasing number of units each layer
    Args:
        input_shape (tuple): Shape of the input layer
        multi (bool): True for model with categorical outpur,
                    False for model with binary output
    Returns:
        Compiled MLP model
    """
    model = Sequential()
    model.add(layers.Dense(units=300, activation='relu', input_shape=input_shape))
    model.add(layers.Dense(units=100, activation='relu'))
    model.add(layers.Dense(units=50, activation='relu'))
    model.add(layers.Dense(units=20, activation='relu'))
    if multi:
        model.add(layers.Dense(6, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy',metrics = ['accuracy'])
    else:
        model.add(layers.Dense(2, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy',metrics = ['accuracy'])

    return model


def cnn_model(input_shape:tuple, multi: bool):
    """Convolution Neural Network model
    (Conv1D -> MaxPooling) x3 -> Conv1d 
    Args:
        input_shape (tuple): Shape of the input layer
        multi (bool): True for model with categorical outpur,
                    False for model with binary output
    Returns:
        Compiled CNN model
    """

    model = Sequential()
    model.add(layers.Conv1D(filters=32, kernel_size=3, input_shape=input_shape))
    model.add(layers.MaxPooling1D(2, padding='same'))
    model.add(layers.Conv1D(filters=16, kernel_size=3))
    model.add(layers.MaxPooling1D(2, padding='same'))
    model.add(layers.Conv1D(filters=32, kernel_size=3))
    model.add(layers.MaxPooling1D(2, padding='same'))
    model.add(layers.Conv1D(filters=64, kernel_size=3))
    model.add(layers.Flatten())
    if multi:
        model.add(layers.Dense(6, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy',metrics = ['accuracy'])
    else:
        model.add(layers.Dense(2, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy',metrics = ['accuracy'])

    return model


def bi_lstm_model(input_shape:tuple, multi: bool):
    """Bi-Directional LSTM model
    Layer of Bidirectional LSTM with two Dropout layers before and after
    Dropout rate - 10% then 50%
    Args:
        input_shape (tuple): Shape of the input layer
        multi (bool): True for model with categorical outpur,
                    False for model with binary output
    Returns:
        Compiled Bi-LSTM model
    """
    model = Sequential()
    model.add(layers.Dense(units=32, activation='relu', input_shape=input_shape))
    model.add(layers.Dropout(0.1))
    model.add(layers.Bidirectional(layers.LSTM(units=20)))
    model.add(layers.Dropout(0.5))
    if multi:
        model.add(layers.Dense(6, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy',metrics = ['accuracy'])
    else:
        model.add(layers.Dense(2, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy',metrics = ['accuracy'])

    return model


def cnn_bi_lstm_model(input_shape:tuple, multi: bool):
    """CNN + Bi-LSTM model
    First CNN layers then Bi-LSTM layers 
    Args:
        input_shape (tuple): Shape of the input layer
        multi (bool): True for model with categorical outpur,
                    False for model with binary output
    Returns:
        Compiled CNN + Bi-LSTM model
    """
    model = Sequential()
    model.add(layers.Conv1D(filters=32, kernel_size=3, input_shape=input_shape))
    model.add(layers.MaxPooling1D(2, padding='same'))
    model.add(layers.Conv1D(filters=16, kernel_size=3))
    model.add(layers.MaxPooling1D(2, padding='same'))
    model.add(layers.Conv1D(filters=32, kernel_size=3))
    model.add(layers.MaxPooling1D(2, padding='same'))
    model.add(layers.Conv1D(filters=64, kernel_size=3))
    model.add(layers.Dropout(0.1))
    model.add(layers.Bidirectional(layers.LSTM(units=20)))
    model.add(layers.Dropout(0.5))
    if multi:
        model.add(layers.Dense(6, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy',metrics = ['accuracy'])
    else:
        model.add(layers.Dense(2, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy',metrics = ['accuracy'])

    return model