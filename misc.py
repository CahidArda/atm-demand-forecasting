import numpy as np
import pickle

def nmae_error(y_actual, y_pred):
    return np.sum(np.absolute(y_pred - y_actual)) / np.sum(y_actual)

def load_pickle(path):
    file = open(path, 'rb')
    d = pickle.load(file)
    file.close()
    return d