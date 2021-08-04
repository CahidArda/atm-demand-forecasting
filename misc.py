import numpy as np

def nmae_error(y_actual, y_pred):
    return np.sum(np.absolute(y_pred - y_actual)) / np.sum(y_actual)