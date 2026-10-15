"""
Feature Engineering Module for Stock Market Prediction Model.
Calculates key financial technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands,
ATR, OBV, Daily Returns, Volatility) and transforms time-series data for ML/DL modeling.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
from config import INDICATOR_PARAMS


def calculate_sma(series: pd.Series, window: int) -> pd.Series:
    """Calculates Simple Moving Average (SMA)."""
    return series.rolling(window=window, min_periods=1).mean()


def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """Calculates Exponential Moving Average (EMA)."""
    return series.ewm(span=span, adjust=False).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculates Relative Strength Index (RSI) using exponential smoothing.
    Oscillates between 0 and 100 (overbought > 70, oversold < 30).
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    # Prevent division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def calculate_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculates Moving Average Convergence Divergence (MACD).
    Returns (macd_line, signal_line, macd_histogram).
    """
    ema_fast = calculate_ema(series, span=fast)
    ema_slow = calculate_ema(series, span=slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, span=signal)
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist


def calculate_bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Calculates Bollinger Bands: (upper_band, middle_band, lower_band, band_width).
    """
    middle_band = series.rolling(window=window, min_periods=1).mean()
    std = series.rolling(window=window, min_periods=1).std().fillna(0)
    upper_band = middle_band + (num_std * std)
    lower_band = middle_band - (num_std * std)
    band_width = (upper_band - lower_band) / middle_band.replace(0, np.nan)
    return upper_band, middle_band, lower_band, band_width.fillna(0)


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculates Average True Range (ATR) to measure price volatility.
    """
    high = df['High']
    low = df['Low']
    close_prev = df['Close'].shift(1)

    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period, min_periods=1).mean()
    return atr


def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """
    Calculates On-Balance Volume (OBV).
    Reflects cumulative volume flow based on closing price movement.
    """
    close_diff = df['Close'].diff().fillna(0)
    direction = np.where(close_diff > 0, 1, np.where(close_diff < 0, -1, 0))
    obv = (df['Volume'] * direction).cumsum()
    return obv


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Augments raw OHLCV stock dataframe with all technical indicators,
    momentum, returns, and lag features.
    """
    df = df.copy()
    close = df['Close']

    # 1. Moving Averages
    df['SMA_20'] = calculate_sma(close, INDICATOR_PARAMS['sma_short'])
    df['SMA_50'] = calculate_sma(close, INDICATOR_PARAMS['sma_medium'])
    df['SMA_200'] = calculate_sma(close, INDICATOR_PARAMS['sma_long'])

    df['EMA_12'] = calculate_ema(close, INDICATOR_PARAMS['ema_short'])
    df['EMA_26'] = calculate_ema(close, INDICATOR_PARAMS['ema_long'])

    # 2. RSI
    df['RSI'] = calculate_rsi(close, INDICATOR_PARAMS['rsi_period'])

    # 3. MACD
    macd, signal, hist = calculate_macd(
        close,
        fast=INDICATOR_PARAMS['macd_fast'],
        slow=INDICATOR_PARAMS['macd_slow'],
        signal=INDICATOR_PARAMS['macd_signal']
    )
    df['MACD'] = macd
    df['MACD_Signal'] = signal
    df['MACD_Hist'] = hist

    # 4. Bollinger Bands
    upper, mid, lower, width = calculate_bollinger_bands(
        close,
        window=INDICATOR_PARAMS['bb_period'],
        num_std=INDICATOR_PARAMS['bb_std']
    )
    df['BB_Upper'] = upper
    df['BB_Middle'] = mid
    df['BB_Lower'] = lower
    df['BB_Width'] = width

    # 5. Volatility & Volume Indicators
    df['ATR'] = calculate_atr(df, INDICATOR_PARAMS['atr_period'])
    df['OBV'] = calculate_obv(df)

    # 6. Returns & Price Momentum
    df['Daily_Return'] = close.pct_change().fillna(0)
    df['Volatility_20'] = df['Daily_Return'].rolling(window=20, min_periods=1).std().fillna(0)
    df['ROC_10'] = ((close - close.shift(10)) / close.shift(10).replace(0, np.nan) * 100).fillna(0)

    # 7. Lag Features
    for lag in [1, 2, 3, 5]:
        df[f'Close_Lag_{lag}'] = close.shift(lag).bfill()

    # Drop any remaining NaN created by rolling operations
    df = df.bfill().ffill()
    return df


def prepare_tabular_data(
    df: pd.DataFrame,
    target_horizon: int = 1
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepares feature matrix X and target y for tabular models (XGBoost, Random Forest).
    target_horizon: Days ahead to predict (1 = next day's close).
    """
    df_feat = add_all_features(df)
    
    # Target is Close price shifted forward by horizon
    y = df_feat['Close'].shift(-target_horizon)
    
    # Feature columns (exclude future targets, date indices are in index)
    feature_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'SMA_20', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26',
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width',
        'ATR', 'Daily_Return', 'Volatility_20', 'ROC_10',
        'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_5'
    ]
    
    # Filter only available columns
    available_cols = [c for c in feature_cols if c in df_feat.columns]
    X = df_feat[available_cols].iloc[:-target_horizon]
    y = y.iloc[:-target_horizon]
    
    return X, y, available_cols
