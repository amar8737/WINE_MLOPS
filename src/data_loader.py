# src/data_loader.py

import pandas as pd
from sklearn.model_selection import train_test_split
import os
from .logger import get_logger # Using relative import

# Use central logger
logger = get_logger(__name__, log_file="data_loader.log")
PROCESSED_DATA_DIR = "data/processed"


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


def split_and_save_data(data: pd.DataFrame, target_col: str = "target", test_size: float = 0.2, random_state: int = 42):
    """
    Split dataset into train and test sets and save them to the processed data folder.
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

    # --- ADDED SAVE LOGIC ---
    # Create the processed data directory if it doesn't exist
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    logger.info(f"Ensured processed data directory exists at: {PROCESSED_DATA_DIR}")

    # Rejoin features and target for saving
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # Define file paths
    train_path = os.path.join(PROCESSED_DATA_DIR, "train.csv")
    test_path = os.path.join(PROCESSED_DATA_DIR, "test.csv")

    # Save to CSV
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.info(f"✅ Train data saved to {train_path}")
    logger.info(f"✅ Test data saved to {test_path}")


if __name__ == "__main__":
    # This block runs when you execute the script directly
    logger.info("Starting data loading and processing script.")
    raw_df = load_data()
    # Assuming the target column is named 'target' in your wine.csv
    # If it's named 'Class', change the column name below
    raw_df.rename(columns={'Wine': 'target'}, inplace=True)
    split_and_save_data(raw_df, target_col="target")
    logger.info("Script finished successfully.")