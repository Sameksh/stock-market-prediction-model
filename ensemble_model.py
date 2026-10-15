"""
Ensemble Model Module for Stock Market Prediction.
Blends predictions from Deep Learning LSTM and Machine Learning XGBoost
using weighted averaging to optimize overall forecasting accuracy and reduce variance.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from config import ENSEMBLE_WEIGHTS, LOOKBACK_WINDOW, TEST_SIZE
from lstm_model import StockLSTMPipeline
from ml_models import StockMLPipeline
from evaluation import evaluate_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class StockEnsemblePipeline:
    """
    Hybrid Ensemble Model combining LSTM and XGBoost.
    - Trains both models on historical stock price sequences & technical indicators.
    - Synchronizes chronological test dates.
    - Calculates optimal weighted ensemble predictions and performance metrics.
    - Produces combined multi-step future price forecasts.
    """

    def __init__(
        self,
        lstm_weight: float = ENSEMBLE_WEIGHTS["lstm"],
        xgb_weight: float = ENSEMBLE_WEIGHTS["xgboost"]
    ):
        total_w = lstm_weight + xgb_weight
        self.lstm_weight = lstm_weight / total_w
        self.xgb_weight = xgb_weight / total_w

        self.lstm_pipeline = StockLSTMPipeline()
        self.xgb_pipeline = StockMLPipeline(model_type="xgboost")

        self.lstm_results = None
        self.xgb_results = None

    def train(self, df: pd.DataFrame, test_size: float = TEST_SIZE) -> Dict[str, Any]:
        """
        Trains both LSTM and XGBoost pipelines, aligns test period predictions,
        and computes the ensemble evaluation metrics.
        """
        logger.info(f"Training Ensemble (LSTM weight: {self.lstm_weight:.2f}, XGB weight: {self.xgb_weight:.2f})...")

        self.lstm_results = self.lstm_pipeline.train(df, test_size=test_size)
        self.xgb_results = self.xgb_pipeline.train(df, test_size=test_size)

        # Synchronize test dates between LSTM and XGBoost
        lstm_dates = self.lstm_results["test_dates"]
        xgb_dates = self.xgb_results["test_dates"]

        # Find intersecting dates
        common_dates = sorted(list(set(lstm_dates).intersection(set(xgb_dates))))

        lstm_date_map = dict(zip(lstm_dates, self.lstm_results["y_test_pred"]))
        xgb_date_map = dict(zip(xgb_dates, self.xgb_results["y_test_pred"]))
        actual_map = dict(zip(lstm_dates, self.lstm_results["y_test_actual"]))

        aligned_actual = np.array([actual_map[d] for d in common_dates])
        aligned_lstm_pred = np.array([lstm_date_map[d] for d in common_dates])
        aligned_xgb_pred = np.array([xgb_date_map[d] for d in common_dates])

        # Compute weighted ensemble prediction
        ensemble_pred = (self.lstm_weight * aligned_lstm_pred) + (self.xgb_weight * aligned_xgb_pred)

        metrics = evaluate_predictions(aligned_actual, ensemble_pred)

        return {
            "model_type": "Ensemble (LSTM + XGBoost)",
            "weights": {"LSTM": self.lstm_weight, "XGBoost": self.xgb_weight},
            "metrics": metrics,
            "test_dates": common_dates,
            "y_test_actual": aligned_actual,
            "y_test_pred": ensemble_pred,
            "lstm_pred": aligned_lstm_pred,
            "xgb_pred": aligned_xgb_pred,
            "lstm_metrics": self.lstm_results["metrics"],
            "xgb_metrics": self.xgb_results["metrics"],
            "feature_importance": self.xgb_results.get("feature_importance", {})
        }

    def predict_future(self, df: pd.DataFrame, days: int = 15) -> Dict[str, List[float]]:
        """
        Generates future forecasts using both models and produces the blended ensemble path.
        """
        lstm_future = self.lstm_pipeline.predict_future(days=days)
        xgb_future = self.xgb_pipeline.predict_future(df, days=days)

        min_len = min(len(lstm_future), len(xgb_future))
        ensemble_future = []
        for i in range(min_len):
            blended = (self.lstm_weight * lstm_future[i]) + (self.xgb_weight * xgb_future[i])
            ensemble_future.append(round(float(blended), 2))

        return {
            "ensemble": ensemble_future,
            "lstm": lstm_future[:min_len],
            "xgboost": xgb_future[:min_len]
        }
