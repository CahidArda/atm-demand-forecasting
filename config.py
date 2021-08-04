config = {
    'load_config': {
        'path': 'DATA_sample_atm.csv',
        'n_atms': 50
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
        'mlp_hidden': [(64, 'relu'), (16, 'relu')]
    },
    'training_config': {
        'learning_rate': 0.2,
        'loss': 'mse',
        'batch_size': 1024,
        'epochs': 5,
        'verbose': 1
    },
    'post_training_config': {
        'save_model_to': None
    }
}