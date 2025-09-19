# src/trainer.py

import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from src.data_loader import load_data, split_data
from src.logger import get_logger
import mlflow
import mlflow.sklearn

# Use central logger
logger = get_logger(__name__, log_file="trainer.log")

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_model(data_path: str = "data/raw/wine.csv", target_col: str = "target"):
    """
    Trains a Logistic Regression model and logs parameters, metrics,
    and the model artifact with MLflow.
    """
    try:
        # Load data
        df = load_data(data_path)
        X_train, X_test, y_train, y_test = split_data(df, target_col=target_col)

        # Start an MLflow run
        with mlflow.start_run():
            logger.info("MLflow run started.")
            
            # Define model parameters
            params = {
                "model_type": "LogisticRegression",
                "scaler": "StandardScaler",
                "max_iter": 500,
                "random_state": 42
            }
            
            # Log parameters
            mlflow.log_params(params)
            logger.info(f"Logged parameters: {params}")

            # Build pipeline
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=params["max_iter"], 
                    random_state=params["random_state"]
                ))
            ])

            # Train
            logger.info("Training model...")
            pipeline.fit(X_train, y_train)

            # Evaluate
            y_pred = pipeline.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            # Log metrics
            mlflow.log_metric("accuracy", acc)
            logger.info(f"Logged metric - Accuracy: {acc:.4f}")

            # Log model artifact
            mlflow.sklearn.log_model(pipeline, "model")
            logger.info("Model logged as an MLflow artifact.")

            # Save the model locally as well
            model_path = os.path.join(MODEL_DIR, "wine_model.pkl")
            joblib.dump(pipeline, model_path)
            logger.info(f"Model saved locally at {model_path}")
            
            print(f"Training complete. Run ID: {mlflow.active_run().info.run_id}")
            return pipeline, acc

    except Exception as e:
        logger.exception(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    train_model()