# src/evaluation.py

import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.data_loader import load_data, split_data
from src.logger import get_logger

# Use our central logger
logger = get_logger(__name__, log_file="evaluation.log")

# Define paths for models and results
MODEL_DIR = "models"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def evaluate_model(
    model_path: str = os.path.join(MODEL_DIR, "wine_model.pkl"),
    data_path: str = "data/raw/wine.csv",
    target_col: str = "target"
):
    """
    Evaluates the trained model on the test set and saves the metrics.
    """
    try:
        # 1. Load Data
        logger.info("Loading data for evaluation...")
        df = load_data(data_path)
        _, X_test, _, y_test = split_data(df, target_col=target_col)

        # 2. Load Model
        if not os.path.exists(model_path):
            logger.error(f"Model not found at {model_path}. Please run the training script first.")
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)

        # 3. Make Predictions
        logger.info("Making predictions on the test set...")
        y_pred = model.predict(X_test)

        # 4. Calculate Metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        logger.info(f"Model Accuracy: {accuracy:.4f}")
        logger.info("Classification Report:")
        logger.info(f"\n{pd.DataFrame(report).transpose()}")
        logger.info("Confusion Matrix:")
        logger.info(f"\n{cm}")

        # 5. Save Metrics to a File
        results_path = os.path.join(RESULTS_DIR, "evaluation_metrics.txt")
        with open(results_path, "w") as f:
            f.write(f"Evaluation Metrics for model: {model_path}\n")
            f.write("="*50 + "\n")
            f.write(f"Accuracy: {accuracy:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(pd.DataFrame(report).transpose().to_string())
            f.write("\n\nConfusion Matrix:\n")
            f.write(str(cm))
        
        logger.info(f"Evaluation metrics saved to {results_path}")
        return accuracy, report, cm

    except Exception as e:
        logger.exception(f"An error occurred during evaluation: {e}")
        raise

if __name__ == "__main__":
    # Run the evaluation when the script is executed directly
    evaluate_model()