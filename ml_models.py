"""
Machine Learning Models Module for Stock Market Prediction.
Implements XGBoost, Random Forest, Linear Regression, and Support Vector Regression (SVR).
Features time-series split, feature scaling, model persistence, and multi-step future forecasting.
"""

import os
import joblib
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb

from config import MODELS_DIR, TEST_SIZE, XGB_CONFIG, RF_CONFIG, RANDOM_STATE
from feature_engineering import prepare_tabular_data, add_all_features
from evaluation import evaluate_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class StockMLPipeline:
    """
    Manages training, evaluation, and future forecasting for tabular machine learning models.
    Supports XGBoost, Random Forest, Linear Regression, and SVR.
    """

    SUPPORTED_MODELS = ["xgboost", "random_forest", "linear_regression", "svr"]

    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type.lower()
        if self.model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model type '{model_type}' not supported. Choose from {self.SUPPORTED_MODELS}")

        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self._init_model()

    def _init_model(self):
        """Instantiates the underlying regression algorithm."""
        if self.model_type == "xgboost":
            self.model = xgb.XGBRegressor(**XGB_CONFIG)
        elif self.model_type == "random_forest":
            self.model = RandomForestRegressor(**RF_CONFIG)
        elif self.model_type == "linear_regression":
            self.model = LinearRegression()
        elif self.model_type == "svr":
            self.model = SVR(kernel="rbf", C=100, gamma=0.01, epsilon=0.1)

    def train(
        self,
        df: pd.DataFrame,
        test_size: float = TEST_SIZE,
        target_horizon: int = 1
    ) -> Dict[str, Any]:
        """
        Trains the selected model on stock data using a strict temporal split.
        Returns evaluation metrics and predictions.
        """
        X, y, feature_names = prepare_tabular_data(df, target_horizon=target_horizon)
        self.feature_names = feature_names

        # Strict chronological split (No random shuffling for time-series)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        logger.info(f"Training {self.model_type.upper()} model on {len(X_train)} samples...")
        if self.model_type == "xgboost":
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_test_scaled, y_test)],
                verbose=False
            )
        else:
            self.model.fit(X_train_scaled, y_train)

        # Predict on train and test
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)

        # Compute evaluation metrics on test set
        metrics = evaluate_predictions(y_test.values, y_test_pred)

        # Feature importances if available
        feature_importance = {}
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            feature_importance = dict(
                sorted(
                    zip(feature_names, [round(float(v), 4) for v in importances]),
                    key=lambda x: x[1],
                    reverse=True
                )
            )

        return {
            "model_type": self.model_type,
            "metrics": metrics,
            "test_dates": list(X_test.index),
            "y_test_actual": y_test.values,
            "y_test_pred": y_test_pred,
            "train_dates": list(X_train.index),
            "y_train_actual": y_train.values,
            "y_train_pred": y_train_pred,
            "feature_importance": feature_importance
        }

    def predict_future(self, df: pd.DataFrame, days: int = 15) -> List[float]:
        """
        Iteratively forecasts future stock closing prices for N days ahead.
        """
        if self.model is None or not self.feature_names:
            raise RuntimeError("Model has not been trained yet.")

        current_df = df.copy()
        future_predictions = []

        for _ in range(days):
            df_with_feats = add_all_features(current_df)
            latest_row = df_with_feats[self.feature_names].iloc[-1:].copy()
            scaled_row = self.scaler.transform(latest_row)
            next_pred = float(self.model.predict(scaled_row)[0])
            future_predictions.append(round(next_pred, 2))

            # Simulate next day row to append for sequential autoregression
            next_date = current_df.index[-1] + pd.Timedelta(days=1)
            # Adjust if weekend
            while next_date.weekday() >= 5:
                next_date += pd.Timedelta(days=1)

            last_close = current_df['Close'].iloc[-1]
            diff = next_pred - last_close
            sim_open = last_close
            sim_close = next_pred
            sim_high = max(sim_open, sim_close) + abs(diff) * 0.2
            sim_low = min(sim_open, sim_close) - abs(diff) * 0.2
            sim_vol = current_df['Volume'].iloc[-5:].mean()

            new_row = pd.DataFrame([{
                'Open': sim_open,
                'High': sim_high,
                'Low': sim_low,
                'Close': sim_close,
                'Volume': sim_vol
            }], index=[next_date])

            current_df = pd.concat([current_df, new_row])

        return future_predictions

    def save(self, filepath: str):
        """Saves model and scaler to disk."""
        data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "model_type": self.model_type
        }
        joblib.dump(data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Loads model and scaler from disk."""
        data = joblib.load(filepath)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]
        self.model_type = data["model_type"]
        logger.info(f"Model loaded from {filepath}")
