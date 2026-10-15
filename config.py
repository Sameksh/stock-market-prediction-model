"""
Configuration module for the Stock Market Prediction Model System.
Contains default parameters, paths, model settings, and technical indicator settings.
"""

import os
from pathlib import Path

# Base Directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "saved_models"
REPORTS_DIR = BASE_DIR / "reports"
OUTPUTS_DIR = BASE_DIR / "outputs"

for directory in [DATA_DIR, MODELS_DIR, REPORTS_DIR, OUTPUTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Default Stock Tickers
DEFAULT_TICKERS = [
    # US Tech & Global Giants
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META",
    # Indian Bluechips (NSE)
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    # Indices & ETFs
    "^GSPC", "^NSEI"
]

DEFAULT_TICKER = "AAPL"
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = None  # None defaults to today

# Feature Engineering / Technical Indicators Parameters
INDICATOR_PARAMS = {
    "sma_short": 20,
    "sma_medium": 50,
    "sma_long": 200,
    "ema_short": 12,
    "ema_long": 26,
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bb_period": 20,
    "bb_std": 2.0,
    "atr_period": 14
}

# Machine Learning & Deep Learning Settings
LOOKBACK_WINDOW = 60          # 60 trading days sequence length for LSTM
TEST_SIZE = 0.2               # 20% validation/test set split
RANDOM_STATE = 42

# LSTM Architecture Parameters
LSTM_CONFIG = {
    "units_layer1": 64,
    "units_layer2": 32,
    "dropout_rate": 0.2,
    "epochs": 25,
    "batch_size": 32,
    "learning_rate": 0.001
}

# XGBoost Parameters
XGB_CONFIG = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE
}

# Random Forest Parameters
RF_CONFIG = {
    "n_estimators": 150,
    "max_depth": 10,
    "min_samples_split": 5,
    "random_state": RANDOM_STATE
}

# Ensemble Weights: Combining LSTM and XGBoost
ENSEMBLE_WEIGHTS = {
    "lstm": 0.55,
    "xgboost": 0.45
}
