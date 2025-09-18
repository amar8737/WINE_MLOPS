import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from src.data_loader import load_data, split_data
from src.logger import get_logger

# Use central logger
logger = get_logger(__name__, log_file="trainer.log")

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_model(data_path: str = "data/raw/wine.csv", target_col: str = "target"):
    """
    Train a Logistic Regression model on the Wine dataset.
    """
    try:
        # Load data
        df = load_data(data_path)
        X_train, X_test, y_train, y_test = split_data(df, target_col=target_col)

        logger.info("Initializing preprocessing + model pipeline")

        # Build pipeline
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, random_state=42))
        ])

        # Train
        logger.info("Training model...")
        pipeline.fit(X_train, y_train)

        # Evaluate
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        logger.info(f"Model trained successfully with Accuracy: {acc:.4f}")

        # Save model
        model_path = os.path.join(MODEL_DIR, "wine_model.pkl")
        joblib.dump(pipeline, model_path)
        logger.info(f"Model saved at {model_path}")

        return pipeline, acc

    except Exception as e:
        logger.exception(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    model, accuracy = train_model()
    print(f"Training completed. Accuracy: {accuracy:.4f}")
