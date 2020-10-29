from model import Model

def init_model():
    # Initialize model
    print('Initializing model...')
    model_config = {
        'dout': True,
        'lr': 1e-4,
        'num_workers': 8,
        'batch_size': 32,
        'restore_iter': 0,
        'total_iter': 20000,
        'model_name': 'NLST-CVD3x2D-Res18',
        'train_source': None,
        'val_source': None,
        'test_source': None
    }
    model_config['save_name'] = '_'.join([
        '{}'.format(model_config['model_name']),
        '{}'.format(model_config['dout']),
        '{}'.format(model_config['lr']),
        '{}'.format(model_config['batch_size']),
    ])

    return Model(**model_config)