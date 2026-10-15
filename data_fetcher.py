"""
Data Fetcher Module for Stock Market Prediction Model.
Retrieves real-time and historical financial data via Yahoo Finance (yfinance),
performs schema sanitization, handles missing values, and implements local caching.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import yfinance as yf

from config import DATA_DIR, DEFAULT_START_DATE, DEFAULT_TICKERS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def clean_stock_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitizes raw dataframe from yfinance:
    - Flattens MultiIndex columns if present
    - Ensures standard columns: Open, High, Low, Close, Volume, (Adj Close)
    - Drops timezone info from index for clean timestamp manipulation
    - Fills forward and backward NaN gaps
    """
    if df.empty:
        return df

    # Flatten multi-level columns if returned by newer yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    # Normalize column names
    col_map = {c: c.capitalize() for c in df.columns}
    df = df.rename(columns=col_map)

    # Standardize 'Adj Close'
    for col in df.columns:
        if 'adj' in col.lower():
            df = df.rename(columns={col: 'Adj Close'})
            break

    # Strip timezone from index
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Ensure required numeric columns
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Forward fill then backward fill any gaps
    df = df.ffill().bfill()
    df = df.dropna(subset=['Close'])
    return df


def fetch_stock_data(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = "1d",
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Fetches historical stock market data for a given ticker symbol.
    Uses local CSV caching to avoid repeated API calls.
    """
    ticker = ticker.strip().upper()
    start_date = start_date or DEFAULT_START_DATE
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")

    cache_file = DATA_DIR / f"{ticker}_{start_date}_{end_date}_{interval}.csv"

    # Check cache
    if use_cache and cache_file.exists():
        logger.info(f"Loading cached stock data from {cache_file.name}")
        try:
            cached_df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if not cached_df.empty:
                return clean_stock_dataframe(cached_df)
        except Exception as e:
            logger.warning(f"Error reading cache file {cache_file}: {e}. Re-fetching.")

    logger.info(f"Downloading live stock data for {ticker} from {start_date} to {end_date}...")
    try:
        yf_ticker = yf.Ticker(ticker)
        df = yf_ticker.history(start=start_date, end=end_date, interval=interval, auto_adjust=False)

        if df.empty:
            logger.warning(f"Ticker.history empty. Attempting yf.download for {ticker}...")
            df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)

        df = clean_stock_dataframe(df)

        if df.empty:
            raise ValueError(f"No price data found for ticker '{ticker}' within specified date range.")

        # Save to local cache
        if use_cache:
            df.to_csv(cache_file)
            logger.info(f"Cached {len(df)} records to {cache_file.name}")

        return df

    except Exception as e:
        logger.error(f"Failed to fetch stock data for {ticker}: {e}")
        # If cache exists even if older, fallback to it
        if cache_file.exists():
            logger.warning("Falling back to existing cache file despite errors.")
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        raise


def get_ticker_info(ticker: str) -> Dict[str, Any]:
    """
    Fetches metadata, summary metrics, and company overview for the ticker.
    """
    ticker = ticker.strip().upper()
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        
        # Extract safe fields
        name = info.get("longName") or info.get("shortName") or ticker
        currency = info.get("currency", "USD")
        current_price = info.get("currentPrice") or info.get("regularMarketPrice") or 0.0
        prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose") or current_price
        
        day_change = current_price - prev_close if current_price and prev_close else 0.0
        day_change_pct = (day_change / prev_close * 100) if prev_close else 0.0
        
        return {
            "ticker": ticker,
            "name": name,
            "currency": currency,
            "current_price": round(float(current_price), 2),
            "previous_close": round(float(prev_close), 2),
            "day_change": round(float(day_change), 2),
            "day_change_pct": round(float(day_change_pct), 2),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh", "N/A"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow", "N/A"),
            "pe_ratio": round(info.get("trailingPE", 0.0), 2) if info.get("trailingPE") else "N/A",
            "market_cap": info.get("marketCap", "N/A"),
            "sector": info.get("sector", "Financial / Technology"),
            "industry": info.get("industry", "Equity"),
            "summary": info.get("longBusinessSummary", "Leading global enterprise traded on public stock exchanges.")
        }
    except Exception as e:
        logger.warning(f"Could not fetch full info for {ticker}: {e}")
        return {
            "ticker": ticker,
            "name": ticker,
            "currency": "USD",
            "current_price": 0.0,
            "previous_close": 0.0,
            "day_change": 0.0,
            "day_change_pct": 0.0,
            "fifty_two_week_high": "N/A",
            "fifty_two_week_low": "N/A",
            "pe_ratio": "N/A",
            "market_cap": "N/A",
            "sector": "Market Equity",
            "industry": "Public Stock",
            "summary": f"Market securities ticker {ticker}."
        }


def get_available_tickers() -> list:
    """Returns the list of recommended stock tickers."""
    return DEFAULT_TICKERS
