"""
Evaluation Module for Stock Market Prediction Model.
Calculates statistical and financial regression metrics:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- R² (Coefficient of Determination)
- Directional Accuracy (Hit rate of predicting upward/downward price movements)
"""

import numpy as np
from typing import Dict, Any, Union
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Mean Absolute Percentage Error (MAPE)."""
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    non_zero = y_true != 0
    if not np.any(non_zero):
        return 0.0
    return float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100.0)


def calculate_directional_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prev: Union[np.ndarray, None] = None
) -> float:
    """
    Calculates Directional Accuracy (% of days correctly predicting Up vs Down).
    If y_prev is provided, direction is computed relative to yesterday's price.
    Otherwise, computed relative to previous element in array.
    """
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    
    if len(y_true) < 2:
        return 100.0

    if y_prev is not None and len(y_prev) == len(y_true):
        actual_dir = np.sign(y_true - y_prev)
        pred_dir = np.sign(y_pred - y_prev)
    else:
        actual_dir = np.sign(y_true[1:] - y_true[:-1])
        pred_dir = np.sign(y_pred[1:] - y_pred[:-1])

    matching = np.sum(actual_dir == pred_dir)
    return float((matching / len(actual_dir)) * 100.0)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prev: Union[np.ndarray, None] = None
) -> Dict[str, Any]:
    """
    Computes a complete suite of regression and trading evaluation metrics.
    """
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mape = calculate_mape(y_true, y_pred)
    r2 = float(r2_score(y_true, y_pred))
    dir_acc = calculate_directional_accuracy(y_true, y_pred, y_prev)

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE (%)": round(mape, 2),
        "R2_Score": round(r2, 4),
        "Directional_Accuracy (%)": round(dir_acc, 2),
        "Mean_Actual": round(float(np.mean(y_true)), 2),
        "Mean_Predicted": round(float(np.mean(y_pred)), 2)
    }
