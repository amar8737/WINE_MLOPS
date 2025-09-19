# src/trainer.py

import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from .logger import get_logger
import mlflow
import mlflow.sklearn
import yaml 

logger = get_logger(__name__, log_file="trainer.log")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_model(
    train_data_path: str = "data/processed/train.csv",
    test_data_path: str = "data/processed/test.csv"
):
    try:
        # --- 1. LOAD PARAMETERS FROM YAML ---
        with open('params.yaml', 'r') as f:
            params = yaml.safe_load(f)
            train_params = params['train'] # Access the 'train' section
        logger.info(f"Loaded parameters: {train_params}")

        # ... (data loading code) ...
        train_df = pd.read_csv(train_data_path)
        test_df = pd.read_csv(test_data_path)
        X_train = train_df.drop(columns=['target'])
        y_train = train_df['target']
        X_test = test_df.drop(columns=['target'])
        y_test = test_df['target']

        mlflow.set_experiment("Wine Classification")

        with mlflow.start_run():
            # --- 2. LOG PARAMETERS TO MLFLOW ---
            # Log the parameters we loaded from the yaml file
            mlflow.log_params(train_params)

            # --- 3. USE PARAMETERS TO BUILD PIPELINE ---
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=train_params["max_iter"], 
                    random_state=train_params["random_state"],
                    C=train_params["C"] 
                ))
            ])

            # ... (the rest of the training, evaluation, and logging code remains the same) ...
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            # Use 'macro' average for multi-class problems
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')
            metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipeline, "model")
            model_path = os.path.join(MODEL_DIR, "wine_model.pkl")
            joblib.dump(pipeline, model_path)
            print(f"Training complete. Run ID: {mlflow.active_run().info.run_id}")
    except Exception as e:
        logger.exception(f"An error occurred during training: {e}")
        raise

if __name__ == "__main__":
    train_model()