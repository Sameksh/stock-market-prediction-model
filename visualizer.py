"""
Visualization Module for Stock Market Prediction Model.
Provides both Matplotlib (publication-ready static figures) and
Plotly (interactive web dashboards, candlestick charts, indicators, and forecast fan charts).
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for scripts & web servers
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import OUTPUTS_DIR


# -------------------------------------------------------------
# Matplotlib Visualizations (Publication / Report Ready)
# -------------------------------------------------------------

def plot_stock_overview_matplotlib(df: pd.DataFrame, ticker: str, save_path: Optional[str] = None) -> str:
    """
    Creates a 3-panel technical chart:
    Panel 1: Price and Moving Averages (SMA 20, 50, 200)
    Panel 2: Trading Volume
    Panel 3: RSI Indicator (with 70/30 bounds)
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [3, 1, 1.2]}, sharex=True)
    fig.patch.set_facecolor("#ffffff")

    # Panel 1: Price
    ax1.plot(df.index, df["Close"], label="Close Price", color="#1f77b4", linewidth=1.5)
    if "SMA_20" in df.columns:
        ax1.plot(df.index, df["SMA_20"], label="SMA 20", color="#ff7f0e", linestyle="--", linewidth=1.2)
    if "SMA_50" in df.columns:
        ax1.plot(df.index, df["SMA_50"], label="SMA 50", color="#2ca02c", linestyle="--", linewidth=1.2)
    if "SMA_200" in df.columns:
        ax1.plot(df.index, df["SMA_200"], label="SMA 200", color="#d62728", linestyle=":", linewidth=1.4)
    ax1.set_title(f"{ticker} - Price History & Moving Averages", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Price ($ / ₹)", fontsize=11)
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Volume
    colors = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(df["Close"], df["Open"])]
    ax2.bar(df.index, df["Volume"], color=colors, alpha=0.7, width=1.0)
    ax2.set_ylabel("Volume", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Panel 3: RSI
    if "RSI" in df.columns:
        ax3.plot(df.index, df["RSI"], color="#9467bd", linewidth=1.4, label="RSI (14)")
        ax3.axhline(70, color="#d62728", linestyle="--", alpha=0.7, label="Overbought (70)")
        ax3.axhline(30, color="#2ca02c", linestyle="--", alpha=0.7, label="Oversold (30)")
        ax3.fill_between(df.index, 30, 70, color="#e0e0e0", alpha=0.4)
        ax3.set_ylabel("RSI", fontsize=11)
        ax3.set_ylim(0, 100)
        ax3.legend(loc="upper left", frameon=True, fontsize=9)
        ax3.grid(True, linestyle="--", alpha=0.5)

    ax3.set_xlabel("Date", fontsize=11)
    plt.tight_layout()

    if not save_path:
        save_path = str(OUTPUTS_DIR / f"{ticker}_overview_matplotlib.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_predictions_matplotlib(
    dates: List[Any],
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    ticker: str,
    metrics: Dict[str, Any],
    save_path: Optional[str] = None
) -> str:
    """
    Plots Actual vs Predicted Stock Prices on Test Data with Metric Scorecard.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#ffffff")

    ax.plot(dates, y_actual, label="Actual Price", color="#111827", linewidth=2.0)
    ax.plot(dates, y_pred, label=f"Predicted Price ({model_name})", color="#2563eb", linewidth=1.8, linestyle="--")

    # Shaded error region
    ax.fill_between(dates, y_actual, y_pred, color="#93c5fd", alpha=0.35, label="Prediction Error Gap")

    ax.set_title(f"{ticker} - Actual vs Predicted Stock Prices ({model_name})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Price ($ / ₹)", fontsize=11)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Metric text box
    metric_str = (
        f"Model: {model_name}\n"
        f"MAE: {metrics.get('MAE', 'N/A')}\n"
        f"RMSE: {metrics.get('RMSE', 'N/A')}\n"
        f"MAPE: {metrics.get('MAPE (%)', 'N/A')}%\n"
        f"R² Score: {metrics.get('R2_Score', 'N/A')}\n"
        f"Dir. Accuracy: {metrics.get('Directional_Accuracy (%)', 'N/A')}%"
    )
    ax.text(
        0.02, 0.25, metric_str,
        transform=ax.transAxes,
        fontsize=10,
        family="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f3f4f6", edgecolor="#cbd5e1", alpha=0.9)
    )

    plt.tight_layout()
    if not save_path:
        save_path = str(OUTPUTS_DIR / f"{ticker}_{model_name}_predictions_matplotlib.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_future_forecast_matplotlib(
    hist_dates: List[Any],
    hist_prices: List[float],
    future_dates: List[Any],
    future_preds: List[float],
    ticker: str,
    model_name: str = "Ensemble",
    save_path: Optional[str] = None
) -> str:
    """
    Plots the connection from recent historical prices into future forecasted days.
    """
    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor("#ffffff")

    # Show last 40 days of history
    lookback_plot = min(50, len(hist_dates))
    ax.plot(hist_dates[-lookback_plot:], hist_prices[-lookback_plot:], label="Historical Prices", color="#1f2937", linewidth=2)

    # Connect bridge point
    bridge_dates = [hist_dates[-1]] + list(future_dates)
    bridge_prices = [hist_prices[-1]] + list(future_preds)
    ax.plot(bridge_dates, bridge_prices, label=f"Forecast ({model_name})", color="#ea580c", linewidth=2.2, linestyle="--", marker="o", markersize=4)

    # Shaded confidence corridor
    upper_band = [p * 1.03 for p in bridge_prices]
    lower_band = [p * 0.97 for p in bridge_prices]
    ax.fill_between(bridge_dates, lower_band, upper_band, color="#fed7aa", alpha=0.4, label="±3% Confidence Corridor")

    ax.set_title(f"{ticker} - Future {len(future_preds)}-Day Price Forecast ({model_name})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Price ($ / ₹)", fontsize=11)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if not save_path:
        save_path = str(OUTPUTS_DIR / f"{ticker}_forecast_matplotlib.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_feature_importance_matplotlib(importances: Dict[str, float], top_n: int = 10, save_path: Optional[str] = None) -> str:
    """Horizontal bar chart for XGBoost / Random Forest feature importance."""
    if not importances:
        return ""
    items = list(importances.items())[:top_n]
    features = [x[0] for x in reversed(items)]
    scores = [x[1] for x in reversed(items)]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(features, scores, color="#3b82f6", edgecolor="#1d4ed8")
    ax.set_title("Top Technical Features (XGBoost)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance Score", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")

    plt.tight_layout()
    if not save_path:
        save_path = str(OUTPUTS_DIR / "feature_importance_matplotlib.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


# -------------------------------------------------------------
# Plotly Visualizations (Interactive Web Dashboard Ready)
# -------------------------------------------------------------

def create_candlestick_chart_plotly(
    df: pd.DataFrame,
    ticker: str,
    show_sma: bool = True,
    show_bollinger: bool = True
) -> go.Figure:
    """
    Creates a rich interactive Plotly Candlestick chart with volume and indicators.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.75, 0.25],
        subplot_titles=(f"{ticker} Interactive Candlestick Chart", "Trading Volume")
    )

    # 1. Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC",
            increasing_line_color="#10b981",
            decreasing_line_color="#ef4444"
        ),
        row=1, col=1
    )

    # Moving Averages
    if show_sma:
        if "SMA_20" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA_20"], mode="lines", name="SMA 20", line=dict(color="#f59e0b", width=1.5)), row=1, col=1)
        if "SMA_50" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA_50"], mode="lines", name="SMA 50", line=dict(color="#3b82f6", width=1.5)), row=1, col=1)
        if "SMA_200" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA_200"], mode="lines", name="SMA 200", line=dict(color="#8b5cf6", width=1.5)), row=1, col=1)

    # Bollinger Bands
    if show_bollinger and "BB_Upper" in df.columns and "BB_Lower" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], mode="lines", name="BB Upper", line=dict(color="#9ca3af", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], mode="lines", name="BB Lower", line=dict(color="#9ca3af", width=1, dash="dot"), fill="tonexty", fillcolor="rgba(156, 163, 175, 0.1)"), row=1, col=1)

    # Volume Bar Chart
    vol_colors = ["#10b981" if c >= o else "#ef4444" for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=vol_colors, opacity=0.7), row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=650,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


def create_prediction_comparison_plotly(
    dates: List[Any],
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    ticker: str
) -> go.Figure:
    """Interactive Actual vs Predicted Plotly Chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=y_actual, mode="lines", name="Actual Close", line=dict(color="#10b981", width=2.5)))
    fig.add_trace(go.Scatter(x=dates, y=y_pred, mode="lines", name=f"Predicted ({model_name})", line=dict(color="#3b82f6", width=2, dash="dash")))

    fig.update_layout(
        title=f"{ticker} - Actual vs Predicted Test Evaluation ({model_name})",
        template="plotly_dark",
        height=480,
        xaxis_title="Date",
        yaxis_title="Price ($ / ₹)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def create_future_forecast_plotly(
    df: pd.DataFrame,
    future_dates: List[Any],
    future_preds: List[float],
    ticker: str,
    model_name: str = "Ensemble"
) -> go.Figure:
    """Interactive multi-step future price forecast chart."""
    recent_df = df.iloc[-40:]
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=recent_df.index, y=recent_df["Close"], mode="lines", name="Recent Close", line=dict(color="#94a3b8", width=2)))

    bridge_x = [recent_df.index[-1]] + list(future_dates)
    bridge_y = [recent_df["Close"].iloc[-1]] + list(future_preds)
    fig.add_trace(go.Scatter(x=bridge_x, y=bridge_y, mode="lines+markers", name=f"Future Forecast ({model_name})", line=dict(color="#f97316", width=2.5), marker=dict(size=6)))

    upper = [y * 1.03 for y in bridge_y]
    lower = [y * 0.97 for y in bridge_y]
    fig.add_trace(go.Scatter(x=bridge_x, y=upper, mode="lines", name="Upper 3% Band", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=bridge_x, y=lower, mode="lines", name="Lower 3% Band", line=dict(width=0), fill="tonexty", fillcolor="rgba(249, 115, 22, 0.15)", showlegend=True))

    fig.update_layout(
        title=f"{ticker} - Next {len(future_preds)} Trading Days Forecast",
        template="plotly_dark",
        height=480,
        xaxis_title="Date",
        yaxis_title="Price ($ / ₹)",
        hovermode="x unified"
    )
    return fig


def create_technical_indicators_plotly(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Plotly subplots for MACD and RSI."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=(f"{ticker} - MACD (12, 26, 9)", f"{ticker} - RSI (14)"))

    # MACD
    if "MACD" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD Line", line=dict(color="#3b82f6")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal Line", line=dict(color="#f97316")), row=1, col=1)
        hist_colors = ["#10b981" if h >= 0 else "#ef4444" for h in df["MACD_Hist"]]
        fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], name="Histogram", marker_color=hist_colors), row=1, col=1)

    # RSI
    if "RSI" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="#a855f7")), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#10b981", row=2, col=1)

    fig.update_layout(template="plotly_dark", height=500, margin=dict(l=40, r=40, t=40, b=30))
    return fig
