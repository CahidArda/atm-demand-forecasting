import numpy as np
import pickle

def nmae_error(y_actual, y_pred):
    return np.sum(np.absolute(y_pred - y_actual)) / np.sum(y_actual)

def load_pickle(path):
    with open(path, 'rb') as file:
        d = pickle.load(file)
    return d

def save_pickle(path, obj):
    with open(path, 'wb') as file:
        pickle.dump(obj, file)

model_path = "/model"

def save_model(path, model, scaler_X, scaler_y):
    model.save_weights(path + model_path)
    save_pickle(path + "/scaler_X.pickle", scaler_X)
    save_pickle(path + "/scaler_y.pickle", scaler_y)

def load_model(path, model):
    # load model
    model.load_weights(path + model_path)   

    # load scalers
    scaler_X = load_pickle(path + "/scaler_X.pickle")
    scaler_y = load_pickle(path + "/scaler_y.pickle")

    return model, scaler_X, scaler_y