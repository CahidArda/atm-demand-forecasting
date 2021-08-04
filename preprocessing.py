
from sklearn.preprocessing import MinMaxScaler

def get_input_sets(feature_set, groups):
    """
    Generate input sets from a feature set for use in my custom TabTransformer

    Args:
        feature_set(:obj:`DataFrame`): Pandas dataframe to generate input sets from.
        groups(:obj:`list`): List of lists with each member list denoting a group of features.

    Returns:
        List of datasets corresponding to the groups
    """
    result = []
    for group in groups:
        result.append(feature_set[group])
    return result

def scaler_fit_transform(X, y, numerical_features):
    scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
    scaler_X.fit(X[numerical_features])
    scaler_y.fit(y.to_numpy().reshape(-1, 1))
    X[numerical_features] = scaler_X.transform(X[numerical_features])
    y = scaler_y.transform(y.to_numpy().reshape(-1, 1)).ravel()
    return X, y, scaler_X, scaler_y

def scaler_transform(X, y, scaler_X, scaler_y, numerical_features):
    X[numerical_features] = scaler_X.transform(X[numerical_features])
    y = scaler_y.transform(y.to_numpy().reshape(-1, 1)).ravel()
    return X, y

def scaler_inverse_transform(y, scaler_y):
    return scaler_y.inverse_transform(y.reshape(-1,1))