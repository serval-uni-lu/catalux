import os
import torch
import random
import numpy as np

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

def df_to_markdown(df, index=False, scaling_factor=None, round_digits=None, str_max_len=None):
    """Converts a DataFrame to a Markdown table.
    """
    df_copy = df.copy()
    
    if scaling_factor is not None:
        numeric_cols = df_copy.select_dtypes(include=np.number).columns
        df_copy[numeric_cols] = df_copy[numeric_cols] * scaling_factor
    
    if round_digits is not None:
        df_copy = df_copy.round(round_digits)
        
    if str_max_len is not None:
        for col in df_copy.select_dtypes(include=['object']):
            df_copy[col] = df_copy[col].apply(
                lambda x: x if len(str(x)) <= str_max_len else str(x)[:str_max_len] + '…'
            )
            
    markdown_str = df_copy.to_markdown(index=index, tablefmt='pipe')
    return markdown_str

def md_print(df, index=False, str_max_len=4):
    print(df_to_markdown(df, index=index, str_max_len=str_max_len))