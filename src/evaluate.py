import os
import sys
import joblib
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as config
from src.preprocess import load_data, preprocess_data

def evaluate_model():
    if not os.path.exists(config.MODEL_PKL_PATH):
        print(f"Error: Trained model not found at {config.MODEL_PKL_PATH}")
        print("Please run train.py first.")
        return

    print("Loading data...")
    df = load_data()
    
    print("Preprocessing data to get the test set...")
    _, X_test, _, y_test, _ = preprocess_data(df)
    
    print(f"Loading trained model from {config.MODEL_PKL_PATH}...")
    pipeline = joblib.load(config.MODEL_PKL_PATH)
    
    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    
    print("\n" + "="*40)
    print("        MODEL EVALUATION RESULTS")
    print("="*40)
    print(f"R² Score: {r2:.4f}")
    print(f"MAE:      {mae:,.2f} EGP")
    print(f"RMSE:     {rmse:,.2f} EGP")
    print(f"MAPE:     {mape:.2f}%")
    print("="*40 + "\n")

if __name__ == "__main__":
    evaluate_model()
