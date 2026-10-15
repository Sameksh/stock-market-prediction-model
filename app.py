"""
Streamlit Web Application: QuantPredict - AI Stock Market Prediction & Analytics Platform.
Features interactive Plotly charts, live yfinance data, real-time LSTM & XGBoost model training,
metric scorecards, technical indicators, and multi-step future forecasting.
"""

import sys
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from config import DEFAULT_TICKERS, DEFAULT_TICKER, DEFAULT_START_DATE, ENSEMBLE_WEIGHTS
from data_fetcher import fetch_stock_data, get_ticker_info
from feature_engineering import add_all_features
from ml_models import StockMLPipeline
from lstm_model import StockLSTMPipeline
from ensemble_model import StockEnsemblePipeline
from visualizer import (
    create_candlestick_chart_plotly,
    create_prediction_comparison_plotly,
    create_future_forecast_plotly,
    create_technical_indicators_plotly
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -------------------------------------------------------------
# Streamlit Page Config & Custom Styling
# -------------------------------------------------------------
st.set_page_config(
    page_title="Stock Market Prediction System | LSTM & XGBoost",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Dark Financial UI Theme
st.markdown("""
<style>
    /* Global Base */
    .stApp {
        background-color: #0b0f19;
        color: #f3f4f6;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Top Header Banner */
    .hero-container {
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 50%, #1e293b 100%);
        border: 1px solid #312e81;
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.0rem;
    }

    /* Metric Cards */
    .metric-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-label {
        color: #9ca3af;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #f9fafb;
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .metric-sub {
        font-size: 0.85rem;
        margin-top: 2px;
    }
    .pos-change { color: #10b981; font-weight: 600; }
    .neg-change { color: #ef4444; font-weight: 600; }

    /* Model Scorecard Container */
    .scorecard-box {
        background: #1e293b;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 12px 0px;
    }
    
    /* Streamlit element overrides */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #1f2937;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        border-radius: 8px 8px 0px 0px;
        font-weight: 600;
        color: #9ca3af;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e293b !important;
        color: #60a5fa !important;
        border-bottom: 2px solid #3b82f6 !important;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# Sidebar Configuration
# -------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/bullish.png", width=64)
st.sidebar.title("Configuration")

# Ticker Selection
ticker_option = st.sidebar.selectbox(
    "Select Stock Preset:",
    options=DEFAULT_TICKERS,
    index=0
)
custom_ticker = st.sidebar.text_input("Or Enter Custom Ticker:", placeholder="e.g. NVDA, MSFT, RELIANCE.NS").strip()
selected_ticker = custom_ticker.upper() if custom_ticker else ticker_option

# Date Range Picker
col_d1, col_d2 = st.sidebar.columns(2)
with col_d1:
    start_date = st.date_input("Start Date", value=datetime(2021, 1, 1))
with col_d2:
    end_date = st.date_input("End Date", value=datetime.now())

# Model Selection
model_choice = st.sidebar.selectbox(
    "Forecasting Algorithm:",
    options=[
        "Ensemble (LSTM + XGBoost)",
        "Deep Learning LSTM",
        "XGBoost Regressor",
        "Random Forest Regressor",
        "Linear Regression",
        "Support Vector Regressor (SVR)"
    ],
    index=0
)

# Lookback Window & Horizon
lookback_days = st.sidebar.slider("Lookback Window (Days):", min_value=30, max_value=120, value=60, step=5)
forecast_days = st.sidebar.slider("Future Forecast Days:", min_value=1, max_value=30, value=15, step=1)

run_button = st.sidebar.button("🚀 Run Prediction & Backtest", type="primary", use_container_width=True)


# -------------------------------------------------------------
# Main Application Content
# -------------------------------------------------------------
try:
    with st.spinner(f"Loading data for {selected_ticker}..."):
        info = get_ticker_info(selected_ticker)
        df_raw = fetch_stock_data(selected_ticker, start_date=str(start_date), end_date=str(end_date))
        df_feat = add_all_features(df_raw)

    # Asset Header Banner
    currency = info["currency"]
    day_change_class = "pos-change" if info["day_change"] >= 0 else "neg-change"
    day_change_symbol = "+" if info["day_change"] >= 0 else ""

    st.markdown(f"""
    <div class="hero-container">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <div class="hero-title">{info['name']} ({selected_ticker})</div>
                <div class="hero-subtitle">{info['sector']} • {info['industry']} • Traded in {currency}</div>
            </div>
            <div style="text-align: right; margin-top: 8px;">
                <span style="font-size: 2.2rem; font-weight: 800;">{currency} {info['current_price']:,.2f}</span>
                <div class="{day_change_class}" style="font-size: 1.05rem;">
                    {day_change_symbol}{info['day_change']:,.2f} ({day_change_symbol}{info['day_change_pct']:.2f}%) Today
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4 Top Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Previous Close</div>
            <div class="metric-value">{currency} {info['previous_close']:,.2f}</div>
            <div class="metric-sub" style="color: #64748b;">Daily settlement</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        high_low = f"{info['fifty_two_week_low']} - {info['fifty_two_week_high']}"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">52-Week Range</div>
            <div class="metric-value" style="font-size: 1.25rem;">{high_low}</div>
            <div class="metric-sub" style="color: #64748b;">Low to High</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        pe_display = str(info['pe_ratio'])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">P/E Ratio</div>
            <div class="metric-value">{pe_display}</div>
            <div class="metric-sub" style="color: #64748b;">Valuation multiple</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Data Points</div>
            <div class="metric-value">{len(df_raw):,}</div>
            <div class="metric-sub" style="color: #64748b;">Trading records</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # ---------------------------------------------------------
    # Main Navigation Tabs
    # ---------------------------------------------------------
    tab_charts, tab_models, tab_forecast, tab_risk, tab_data = st.tabs([
        "📊 Interactive Charts & Indicators",
        "🧠 Model Training & Evaluation",
        "🔮 Future Price Forecast",
        "🛡️ Risk & Volatility Analytics",
        "📋 Historical Data Table"
    ])

    # ---------------------------------------------------------
    # Tab 1: Charts & Technical Indicators
    # ---------------------------------------------------------
    with tab_charts:
        st.subheader("Price Action & Technical Analysis")
        col_t1, col_t2 = st.columns([1, 1])
        with col_t1:
            show_sma = st.checkbox("Overlay Moving Averages (SMA 20, 50, 200)", value=True)
        with col_t2:
            show_bb = st.checkbox("Overlay Bollinger Bands (20, 2.0σ)", value=True)

        fig_candle = create_candlestick_chart_plotly(df_feat, selected_ticker, show_sma=show_sma, show_bollinger=show_bb)
        st.plotly_chart(fig_candle, use_container_width=True)

        st.subheader("Momentum Oscillators (MACD & RSI)")
        fig_tech = create_technical_indicators_plotly(df_feat, selected_ticker)
        st.plotly_chart(fig_tech, use_container_width=True)

    # ---------------------------------------------------------
    # Tab 2: Model Training & Evaluation
    # ---------------------------------------------------------
    with tab_models:
        st.subheader(f"Model Training & Backtest Evaluation: {model_choice}")
        
        # Check cache or state
        state_key = f"results_{selected_ticker}_{model_choice}_{len(df_feat)}_{lookback_days}_{forecast_days}"
        
        if state_key not in st.session_state or run_button:
            with st.spinner(f"Training {model_choice} model on historical records..."):
                if "Ensemble" in model_choice:
                    pipeline = StockEnsemblePipeline()
                    results = pipeline.train(df_feat)
                    future_dict = pipeline.predict_future(df_raw, days=forecast_days)
                    results["future_preds"] = future_dict["ensemble"]
                elif "LSTM" in model_choice:
                    pipeline = StockLSTMPipeline(lookback_window=lookback_days)
                    results = pipeline.train(df_feat)
                    results["future_preds"] = pipeline.predict_future(days=forecast_days)
                elif "XGBoost" in model_choice:
                    pipeline = StockMLPipeline(model_type="xgboost")
                    results = pipeline.train(df_raw)
                    results["future_preds"] = pipeline.predict_future(df_raw, days=forecast_days)
                elif "Random Forest" in model_choice:
                    pipeline = StockMLPipeline(model_type="random_forest")
                    results = pipeline.train(df_raw)
                    results["future_preds"] = pipeline.predict_future(df_raw, days=forecast_days)
                elif "Linear" in model_choice:
                    pipeline = StockMLPipeline(model_type="linear_regression")
                    results = pipeline.train(df_raw)
                    results["future_preds"] = pipeline.predict_future(df_raw, days=forecast_days)
                else:
                    pipeline = StockMLPipeline(model_type="svr")
                    results = pipeline.train(df_raw)
                    results["future_preds"] = pipeline.predict_future(df_raw, days=forecast_days)

                st.session_state[state_key] = results

        results = st.session_state[state_key]
        metrics = results["metrics"]

        # 5 Metric KPI Cards
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("MAE", f"{metrics['MAE']:.3f}", help="Mean Absolute Error in stock currency")
        with m2:
            st.metric("RMSE", f"{metrics['RMSE']:.3f}", help="Root Mean Squared Error (penalizes large errors)")
        with m3:
            st.metric("MAPE", f"{metrics['MAPE (%)']:.2f}%", help="Mean Absolute Percentage Error")
        with m4:
            st.metric("R² Score", f"{metrics['R2_Score']:.4f}", help="Variance explained by the model")
        with m5:
            st.metric("Directional Hit Rate", f"{metrics['Directional_Accuracy (%)']:.1f}%", help="Percentage of days correctly predicting price direction")

        # Actual vs Predicted Plot
        fig_pred = create_prediction_comparison_plotly(
            results["test_dates"],
            results["y_test_actual"],
            results["y_test_pred"],
            model_choice,
            selected_ticker
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        # Feature Importance if tree-based
        if "feature_importance" in results and results["feature_importance"]:
            st.subheader("Feature Importance Analysis")
            fi = results["feature_importance"]
            top_fi = dict(list(fi.items())[:12])
            fi_df = pd.DataFrame({"Feature": list(top_fi.keys()), "Importance": list(top_fi.values())})
            fi_fig = go.Figure(go.Bar(
                x=fi_df["Importance"],
                y=fi_df["Feature"],
                orientation="h",
                marker=dict(color="#3b82f6")
            ))
            fi_fig.update_layout(template="plotly_dark", height=380, title="Top Predictive Indicators", yaxis=dict(autorange="reversed"))
            st.plotly_chart(fi_fig, use_container_width=True)

    # ---------------------------------------------------------
    # Tab 3: Future Price Forecast
    # ---------------------------------------------------------
    with tab_forecast:
        st.subheader(f"Next {forecast_days}-Day Forecast Projection")
        future_preds = results.get("future_preds", [])
        
        # Calculate future trading business days
        last_date = df_raw.index[-1]
        future_dates = []
        curr = last_date
        while len(future_dates) < len(future_preds):
            curr += pd.Timedelta(days=1)
            if curr.weekday() < 5:
                future_dates.append(curr)

        fig_fut = create_future_forecast_plotly(
            df_raw,
            future_dates,
            future_preds[:len(future_dates)],
            selected_ticker,
            model_name=model_choice
        )
        st.plotly_chart(fig_fut, use_container_width=True)

        # Forecast Table
        curr_price = float(df_raw["Close"].iloc[-1])
        table_rows = []
        for d, p in zip(future_dates, future_preds):
            diff = p - curr_price
            diff_pct = (diff / curr_price) * 100
            signal = "🟢 Bullish" if diff > 0 else "🔴 Bearish" if diff < 0 else "⚪ Neutral"
            table_rows.append({
                "Forecast Date": d.strftime("%Y-%m-%d (%a)"),
                f"Predicted Price ({currency})": f"{p:,.2f}",
                f"Change from Current ({currency})": f"{'+' if diff >= 0 else ''}{diff:,.2f}",
                "Expected Return (%)": f"{'+' if diff_pct >= 0 else ''}{diff_pct:.2f}%",
                "Market Signal": signal
            })

        forecast_table_df = pd.DataFrame(table_rows)
        st.dataframe(forecast_table_df, use_container_width=True)

        # Download CSV button
        csv_data = forecast_table_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv_data,
            file_name=f"{selected_ticker}_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

    # ---------------------------------------------------------
    # Tab 4: Risk & Volatility
    # ---------------------------------------------------------
    with tab_risk:
        st.subheader("Asset Risk & Portfolio Volatility Analysis")
        returns = df_feat["Daily_Return"]
        vol_20 = float(df_feat["Volatility_20"].iloc[-1] * 100)
        ann_vol = float(vol_20 * np.sqrt(252))
        var_95 = float(np.percentile(returns, 5) * 100)

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.metric("20-Day Volatility", f"{vol_20:.2f}%", help="Standard deviation of recent daily returns")
        with r2:
            st.metric("Annualized Volatility", f"{ann_vol:.2f}%", help="Annualized price fluctuation (252 trading days)")
        with r3:
            st.metric("Daily 95% VaR", f"{var_95:.2f}%", help="Value at Risk: maximum expected single-day loss with 95% confidence")
        with r4:
            sharpe_est = (returns.mean() / (returns.std() + 1e-8)) * np.sqrt(252)
            st.metric("Sharpe Ratio Est.", f"{sharpe_est:.2f}", help="Risk-adjusted return benchmark")

        # Return distribution plot
        ret_fig = go.Figure()
        ret_fig.add_trace(go.Histogram(x=returns * 100, nbinsx=60, marker_color="#38bdf8", opacity=0.75, name="Daily Returns"))
        ret_fig.update_layout(
            template="plotly_dark",
            height=380,
            title=f"{selected_ticker} - Daily Return Distribution (%)",
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency"
        )
        st.plotly_chart(ret_fig, use_container_width=True)

    # ---------------------------------------------------------
    # Tab 5: Raw Dataset
    # ---------------------------------------------------------
    with tab_data:
        st.subheader("Historical Stock Data & Technical Features")
        st.write(f"Displaying latest 100 records for **{selected_ticker}**:")
        st.dataframe(df_feat.tail(100), use_container_width=True)

except Exception as ex:
    st.error(f"Error loading stock data or running pipeline: {ex}")
    st.info("Tip: Check ticker spelling, network connection, or adjust date range.")
