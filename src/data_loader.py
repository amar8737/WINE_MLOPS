import pandas as pd
from sklearn.model_selection import train_test_split
import os
from src.logger import get_logger

# Use central logger
logger = get_logger(__name__, log_file="data_loader.log")

def load_data(data_path: str = "data/raw/wine.csv"):
    """
    Load the wine dataset from a CSV file.
    """
    if not os.path.exists(data_path):
        logger.error(f"Dataset not found at {data_path}")
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    logger.info(f"Loading dataset from {data_path}")
    data = pd.read_csv(data_path)
    logger.info(f"Dataset loaded successfully with shape {data.shape}")
    return data


def split_data(data: pd.DataFrame, target_col: str = "target", test_size: float = 0.2, random_state: int = 42):
    """
    Split dataset into train and test sets.
    """
    if target_col not in data.columns:
        logger.error(f"Target column '{target_col}' not found in dataset")
        raise KeyError(f"Target column '{target_col}' not found in dataset")
    
    X = data.drop(columns=[target_col])
    y = data[target_col]

    logger.info(f"Splitting dataset (test_size={test_size}, random_state={random_state})")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(
        f"Data split completed → Train shape: {X_train.shape}, Test shape: {X_test.shape}"
    )

    return X_train, X_test, y_train, y_test
