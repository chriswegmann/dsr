{
    'dataset_loader_train': {
        '__factory__': 'dataset.OpenML',
        'name': 'wine-quality-red',
    },

    'model_persister': {
        '__factory__': 'palladium.persistence.Database',
        'url': 'sqlite:///model.db',
    },

    'model': {
        '__factory__': 'model.create_pipeline',
        # YOUR CODE HERE:
        #
        # The new dataset has a different number of inputs and
        # outputs.  Adjust these parameters:
        #
        'module__num_inputs': 11,   # Number of features
        'module__num_outputs': 1,  # Number of classes
    },

    'scoring': 'neg_mean_absolute_error',

    'grid_search': {
        'param_grid': {
            'net__lr': [0.1],
            'net__max_epochs': [200],
            'net__module__num_units': [5, 10, 20],
        },
        'cv': 5,
        'verbose': 4,
        'n_jobs': -1,
    },
}
