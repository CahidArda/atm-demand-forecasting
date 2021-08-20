config = {
    'load_config': {
        'data_path': '../internship-study/notebooks/atm_demand/DATA_sample_atm.csv',
        'hyperparameter_path': 'hyperparemeter.txt',
        'n_atms': 100,
        'clusters': {
            'Day_of_the_Week_Index_ClusterId': {
                'path': 'pickles/Day_of_the_Week_Index_cluster_pickle',
                'n_clusters': 20
            },
            'Month_of_the_Year_Index_ClusterId': {
                'path': 'pickles/Month_of_the_Year_Index_cluster_pickle',
                'n_clusters': 20
            },
            'Special_Lunar_Dates_Index_ClusterId': {
                'path': 'pickles/Special_Lunar_Dates_Index_cluster_pickle',
                'n_clusters': 20
            }
        }
    },
    'feature_config': {
        'target': 'CashIn',
        'categorical_column_types': ['int8', 'int64'],
        'excluded_categorical': ['AtmId', 'Day_Index_0', 'Day_Index_1', 'Day_Index_2',
            'Day_Index_3', 'Day_Index_4', 'Day_Index_5', 'Day_Index_6', 'Is_Weekend',
            'curr_month_1_delta', 'curr_month_15_delta', 'next_month_1_delta',
            'is_ramazan', 'ramazan_in_7_days', 'is_kurban', 'kurban_in_7_days',
            'is_cocuk_bayrami', 'is_isci_bayrami', 'is_spor_bayrami',
            'is_zafer_bayrami', 'is_cumhuriyet_bayrami'],
        
        'continuous_column_types': ['float64'],
        'excluded_continuous': ['CashIn', 'CashOut']
    },
    'model_config': {
        'dim': 16,
        'dim_out': 1,
        'depth': 6,
        'heads': 8,
        'attn_dropout': 0.1,
        'ff_dropout': 0.1,
        'mlp_hidden': [(64, 'relu'), (16, 'relu')],
        'normalize_continuous': True
    },
    'training_config': {
        'learning_rate': 0.05,
        'loss': 'mse',
        'batch_size': 1024,
        'epochs': 5,
        'verbose': 1
    },
    'do': 'predict',
    'fit_predict': {
        'save_model_to': './model'
    },
    'predict': {
        'load_model_from': './model'
    }
}

def read_hyperparameters_from_file(path):
    import json

    with open(path) as json_file:
        params = json.load(json_file)

        model_config = config['model_config']
        
        model_config['dim'] = params['dim']
        model_config['depth'] = params['depth']
        model_config['heads'] = params['heads']
        model_config['attn_dropout'] = params['attn_dropout']
        model_config['ff_dropout'] = params['ff_dropout']

        act = params['mlp_activation']
        model_config['mlp_hidden'] = [(params['mlp_1_dim'], act), (params['mlp_2_dim'], act)]

        config['training_config']['learning_rate'] = params['learning_rate']

    return config