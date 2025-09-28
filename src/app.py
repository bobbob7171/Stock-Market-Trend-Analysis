# app.py

import streamlit as st
import pandas as pd
import yfinance as yf
from src.data_handler import (
    fetch_stock_data, clean_all_tickers, align_with_trading_days,
    detect_outliers, data_quality_report, validate_preprocessing,
    pivot_wide, save_outputs, load_csv, validate_schema
)
from src.analytics import calculate_daily_returns, calculate_sma, detect_streaks
from src.profit import calculate_profits
from src.graph import plot_price_sma, plot_streaks, plot_profit_comparison
from config import CONFIG

st.set_page_config(page_title="Stock Analytics Dashboard", layout="wide")
st.title("📈 Stock Analytics Dashboard")
st.markdown("Analyze stocks, visualize trends, calculate profits, and export data.")

# -------------------------
# Sidebar Inputs
# -------------------------
st.sidebar.header("Data Settings")
use_csv = st.sidebar.checkbox("Load data from CSV instead of yfinance?", value=False)

tickers_input = st.sidebar.text_input(
    "Tickers (comma-separated)",
    value="AAPL,MSFT,TSLA"  # default example tickers
)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

start_date = st.sidebar.date_input("Start Date", pd.to_datetime(CONFIG["start_date"]))
end_date = st.sidebar.date_input("End Date", pd.to_datetime(CONFIG["end_date"]))

sma_input = st.sidebar.text_input(
    "SMA Windows (comma-separated)",
    value=",".join(map(str, CONFIG["sma_windows"]))
)
sma_windows = [int(x.strip()) for x in sma_input.split(",") if x.strip().isdigit() and int(x) > 0]

# -------------------------
# Data Loading
# -------------------------
if use_csv:
    csv_path = st.sidebar.text_input("CSV Path", CONFIG["backup_path"])
    cleaned = load_csv(csv_path)
    validate_schema(cleaned)
else:
    # Optional: validate tickers before fetching
    valid_tickers = []
    for t in tickers:
        try:
            test_data = yf.Ticker(t).history(period="1d")
            if not test_data.empty:
                valid_tickers.append(t)
            else:
                st.warning(f"Ticker {t} returned no data.")
        except Exception:
            st.warning(f"Ticker {t} is invalid or cannot be fetched.")
    tickers = valid_tickers

    if tickers:
        cleaned_raw = fetch_stock_data(tickers, str(start_date), str(end_date), CONFIG["backup_path"])
        cleaned = clean_all_tickers(cleaned_raw, tickers)
        cleaned = align_with_trading_days(cleaned, str(start_date), str(end_date))
        cleaned = detect_outliers(cleaned)
        validate_schema(cleaned)
        validate_preprocessing(cleaned, str(start_date), str(end_date))
        save_outputs(cleaned, pivot_wide(cleaned), CONFIG)
    else:
        st.error("No valid tickers to fetch. Please check your input.")
        st.stop()

# -------------------------
# Analytics
# -------------------------
analytics_data = calculate_daily_returns(cleaned)
analytics_data = calculate_sma(analytics_data, sma_windows)
analytics_data = detect_streaks(analytics_data)

# -------------------------
# Profit Calculations
# -------------------------
profit_summary = calculate_profits(analytics_data, tickers)

# -------------------------
# Tabs
# -------------------------
tabs = st.tabs(["Data Quality", "Analytics & Streaks", "Charts", "Profits"])

# Tab 1: Data Quality
with tabs[0]:
    st.subheader("📝 Data Quality Report")
    dq = data_quality_report(cleaned)
    st.dataframe(dq)

    if not cleaned.empty:
        st.download_button(
            "💾 Download Cleaned Data",
            cleaned.to_csv(index=False).encode(),
            "cleaned_data.csv"
        )

# Tab 2: Analytics & Streaks
with tabs[1]:
    st.subheader("📊 Analytics Summary")
    st.dataframe(analytics_data.head())

    st.subheader("📌 Streak Summary")
    streak_summary = pd.DataFrame([
        {
            "Ticker": t,
            "Highest_Up_Streak": grp["Streak"].loc[grp["Streak"]>0].max() if not grp.empty else 0,
            "Highest_Down_Streak": grp["Streak"].loc[grp["Streak"]<0].min() if not grp.empty else 0,
            "Total_Up_Days": (grp["Daily_Return"]>0).sum(),
            "Total_Down_Days": (grp["Daily_Return"]<0).sum()
        }
        for t, grp in analytics_data.groupby("Ticker")
    ])
    st.dataframe(streak_summary)

    # Downloads
    if not analytics_data.empty:
        st.download_button(
            "💾 Download Analytics Data",
            analytics_data.to_csv(index=False).encode(),
            "analytics_data.csv"
        )
    if not streak_summary.empty:
        st.download_button(
            "💾 Download Streak Summary",
            streak_summary.to_csv(index=False).encode(),
            "streak_summary.csv"
        )

# Tab 3: Charts
with tabs[2]:
    st.subheader("📈 Price Charts with SMAs & Streaks")
    selected_ticker = st.selectbox("Select Ticker", tickers)
    st.write("Price + SMA Chart")
    plot_price_sma(analytics_data, selected_ticker, sma_windows=sma_windows)
    st.write("Streaks Chart")
    plot_streaks(analytics_data, selected_ticker)

    if not analytics_data.empty:
        st.download_button(
            "💾 Download Chart Data (Analytics)",
            analytics_data.to_csv(index=False).encode(),
            "chart_data.csv"
        )

# Tab 4: Profits
with tabs[3]:
    st.subheader("💰 Profit Summary")
    st.dataframe(profit_summary)
    st.write("Profit Comparison Chart")
    plot_profit_comparison(profit_summary)

    if not profit_summary.empty:
        st.download_button(
            "💾 Download Profit Summary",
            profit_summary.to_csv(index=False).encode(),
            "profit_summary.csv"
        )
