from pathlib import Path

__all__ = ['project_path', 'dataset_config']

project_path = Path(__file__).parent


dataset_config = {
    'Yelp':
        {
            'hidden_channels': 96,
            'lr': 0.001,
            'weight_decay': 1e-5,
            'beta': 1.,
            'epochs': 50,
        },
    'Elliptic':
        {
            'hidden_channels': 96,
            'lr': 0.01,
            'weight_decay': 1e-5,
            'beta': 1.,
            'epochs': 50
        },
    'Amazon':
        {
            'hidden_channels': 96,
            'lr': 0.01,
            'weight_decay': 1e-5,
            'beta': 1.,
            'epochs': 100,
        },
    'TFinance':
        {
            'hidden_channels': 96,
            'lr': 0.01,
            'weight_decay': 1e-5,
            'beta': 1.,
            'epochs': 50,
        },
    'TSocial':
        {
            'hidden_channels': 96,
            'lr': 0.01,
            'weight_decay': 1e-5,
            'beta': 1.,
            'epochs': 50,
        },

}