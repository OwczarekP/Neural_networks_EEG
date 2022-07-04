import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np


def normalize_data(input_data: np.ndarray):
    norm = np.linalg.norm(input_data)
    normalized = input_data/norm
    return normalized