
from config import read_hyperparameters_from_file, config
from feature_generation import get_all_atms_feature_set
from preprocessing import get_input_sets, scaler_fit_transform, scaler_transform, scaler_inverse_transform
from tabTransformer import TabTransformer
from misc import nmae_error, load_pickle

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# Loading Data
# --------------------------------------------------

load_config = config['load_config']

try:
    config = read_hyperparameters_from_file(load_config['hyperparameter_path'])
except:
    print("WARNING: Hyperparameter file (%s) not found. Using the default config." % load_config['hyperparameter_path'])

clusters = load_config['clusters']

df = pd.read_csv(load_config['data_path'])
all_atms_feature_set = get_all_atms_feature_set(df, first_n = load_config['n_atms'])
all_atms_feature_set.sort_index(inplace = True)

# Reading Pickles
for cluster_feature in clusters:
    d = load_pickle(clusters[cluster_feature]['path'])
    all_atms_feature_set[cluster_feature] = all_atms_feature_set['AtmId'].map(d)

# --------------------------------------------------
# Setting Features
# --------------------------------------------------

feature_config  = config['feature_config']

categorical_features = [cat for cat in
    all_atms_feature_set.select_dtypes(include=feature_config['categorical_column_types'])
    if cat not in feature_config['excluded_categorical']]
continuous_features = [cat for cat in
    all_atms_feature_set.select_dtypes(include=feature_config['continuous_column_types'])
    if cat not in feature_config['excluded_continuous']]

groups = [continuous_features]
groups.extend(categorical_features)

# --------------------------------------------------
# Aranging train/test Data
# --------------------------------------------------

X = all_atms_feature_set[continuous_features + categorical_features]
y = all_atms_feature_set[feature_config['target']]

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

# MinMaxTransform
X_train, y_train, scaler_X, scaler_y = scaler_fit_transform(X_train, y_train, continuous_features)
X_test, y_test = scaler_transform(X_test, y_test, scaler_X, scaler_y, continuous_features)

X_train = get_input_sets(X_train, groups)
X_test  = get_input_sets(X_test, groups)

# --------------------------------------------------
# TabTransformer Model
# --------------------------------------------------

model_config = config['model_config']

tabTransformer = TabTransformer(
    categories = [len(all_atms_feature_set[cat].unique())
        if cat not in clusters.keys() else
        clusters[cat]['n_clusters']
        for cat in categorical_features],
    num_continuous = len(continuous_features),
    dim = model_config['dim'],
    dim_out = model_config['dim_out'],
    depth = model_config['depth'],
    heads = model_config['heads'],
    attn_dropout = model_config['attn_dropout'],
    ff_dropout = model_config['ff_dropout'],
    mlp_hidden = model_config['mlp_hidden']
)

# --------------------------------------------------
# Training
# --------------------------------------------------

training_config = config['training_config']

tabTransformer.compile(
    optimizer = tf.optimizers.Adam(learning_rate = training_config['learning_rate']),
    loss = training_config['loss']
)

history = tabTransformer.fit(X_train,
    y_train,
    batch_size = training_config['batch_size'],
    epochs = training_config['epochs'],
    validation_data = (X_test, y_test),
    verbose = training_config['verbose'])

# --------------------------------------------------
# Post Training
# --------------------------------------------------

post_training_config = config['post_training_config']

save_model_to = post_training_config['save_model_to']
if save_model_to != None:
    tabTransformer.save_weights(save_model_to)

print("Train score: %.4f, test score: %.4f" % 
    (nmae_error(scaler_inverse_transform(y_train, scaler_y), scaler_y.inverse_transform(tabTransformer.predict(X_train))),
    nmae_error(scaler_inverse_transform(y_test, scaler_y), scaler_y.inverse_transform(tabTransformer.predict(X_test)))))