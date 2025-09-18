# src/data_loader.py

import pandas as pd
from sklearn.model_selection import train_test_split
import os
import logging

# ----------------------------
# Setup logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,  # Can change to DEBUG for more details
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/data_loader.log"),  # Save logs to file
        logging.StreamHandler()  # Also print to console
    ]
)

logger = logging.getLogger(__name__)


def load_data(data_path: str = "data/raw/wine.csv"):
    """
    Load the wine dataset from a CSV file.
    """
    if not os.path.exists(data_path):
        logger.error(f"❌ Dataset not found at {data_path}")
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    logger.info(f"📥 Loading dataset from {data_path}")
    data = pd.read_csv(data_path)
    logger.info(f"✅ Dataset loaded successfully with shape {data.shape}")
    return data


def split_data(data: pd.DataFrame, target_col: str = "target", test_size: float = 0.2, random_state: int = 42):
    """
    Split dataset into train and test sets.
    """
    if target_col not in data.columns:
        logger.error(f"❌ Target column '{target_col}' not found in dataset")
        raise KeyError(f"Target column '{target_col}' not found in dataset")
    
    X = data.drop(columns=[target_col])
    y = data[target_col]

    logger.info(f"📊 Splitting dataset (test_size={test_size}, random_state={random_state})")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(
        f"✅ Data split completed → Train shape: {X_train.shape}, Test shape: {X_test.shape}"
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    try:
        df = load_data()
        X_train, X_test, y_train, y_test = split_data(df, target_col="target")
    except Exception as e:
        logger.exception("❌ Error occurred while processing data")
