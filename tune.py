
import random
import logging

import wandb

logging.basicConfig(filename='hyperparameter_sweep.log', level=logging.INFO, format='%(asctime)s - %(message)s')

sweep_config = {
    'method': 'random',  # or 'grid' or 'bayes'
    'metric': {
        'name': 'accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'epochs': {
            'values': list(range(50, 200, 10))
        },
        'new_adj_dim': {
            'values': list(range(5, 15))
        },
        'pretrain_epochs': {
            'values': list(range(30, 100, 10))
        },
        'numSampledCCs': {
            'values': list(range(4, 18))
        },
        'numSampledCycles': {
            'values': list(range(6, 60))
        }
    }
}

sweep_id = wandb.sweep(sweep=sweep_config, project='hyperparameter-sweep')
