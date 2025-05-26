import os
import yaml
import torch
import random
import numpy as np

from loguru import logger

def load_config(config_path='config.yaml'):
    """Load configuration from a YAML file.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if config.get('device') == 'auto':
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif config.get('device') not in ['cuda', 'cpu']:
        logger.warning(f"Warning: Invalid device '{config.get('device')}' specified in config. Using 'cpu'.")
        config['device'] = 'cpu'

    config['lr'] = float(config['lr'])
    config['weight_decay'] = float(config['weight_decay'])
    
    return config

def seed_everything(seed):
    """Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
