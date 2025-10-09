"""
main.py

Streamlit dashboard for Stock Market Trend Analysis:
- Load data from CSV or Yahoo Finance
- Clean, validate, and process stock data
- Calculate daily returns, SMA, streaks, risk-return, and profits
- Visualize trends, charts, and summary metrics
- Enable downloads of CSVs and Plotly charts
"""

import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
from src.data_handler import (
    fetch_stock_data, clean_all_tickers, align_with_trading_days,
    detect_outliers, data_quality_report, validate_preprocessing,
    pivot_wide, save_outputs, load_csv_files, validate_schema
)
from src.analytics import (
    calculate_daily_returns, calculate_sma, detect_streaks, calculate_annual_risk_return
)
from src.profit import calculate_profits
from src.graph import plot_price_sma, plot_streaks, plot_profit_comparison, plot_annual_risk_return, plot_best_buy_sell


# Helper Functions
def format_df(df, fmt="{:.2f}"):
    return df.style.format({col: fmt for col in df.select_dtypes(include="float").columns})

def validate_tickers(tickers: list) -> list:
    valid = []
    for t in tickers:
        try:
            data = yf.Ticker(t).history(period="1d")
            if not data.empty:
                valid.append(t)
            else:
                st.warning(f"Ticker {t} returned no data.")
        except Exception:
            st.warning(f"Ticker {t} is invalid or cannot be fetched.")
    return valid


# Page Setup
st.set_page_config(page_title="Stock Analytics Dashboard", layout="wide")
st.title("📈 Stock Analytics Dashboard")
st.markdown("Analyze stocks, visualize trends, calculate profits, and export data.")

# Sidebar: Data Source
st.sidebar.header("Data Settings")
data_source = st.sidebar.radio(
    "Select Data Source:",
    options=["Upload CSV", "Fetch from yfinance"]
)

tickers = []
ticker_settings = {}
all_cleaned = pd.DataFrame()
today = datetime.date.today()

# Global Settings with Override
with st.sidebar.expander("🌐 Global Settings (Override All)", expanded=False):
    use_global = st.checkbox("Use Global Settings for All Tickers", value=False)
    
    global_start = st.date_input("Global Start Date", value=today - datetime.timedelta(days=90))
    global_end = st.date_input("Global End Date", value=today)
    
    global_sma_input = st.text_input("Global SMA Windows (comma-separated)", value="")
    # Parse SMA if valid
    global_sma_windows = [int(x.strip()) for x in global_sma_input.split(",") if x.strip().isdigit() and int(x) > 0]

# Load Data: CSV
if data_source == "Upload CSV":
    uploaded_files = st.sidebar.file_uploader(
        "Choose CSV file(s)", type=["csv"], accept_multiple_files=True
    )
    if not uploaded_files:
        st.info("Please upload at least one CSV file to continue.")
        st.stop()

    cleaned, tickers = load_csv_files(uploaded_files)
    if cleaned.empty or not tickers:
        st.error("Uploaded CSV(s) did not contain valid data or tickers.")
        st.stop()
    st.toast(f"Loaded {len(uploaded_files)} file(s) successfully: {', '.join(tickers)}")

    st.sidebar.markdown("### Per-Ticker Settings")
    for t in tickers:
        ticker_df = cleaned[cleaned["Ticker"] == t]
        min_date, max_date = ticker_df['Date'].min().date(), ticker_df['Date'].max().date()
        
        with st.sidebar.expander(f"📊 {t}", expanded=not use_global):
            if use_global:
                st.info("🌐 Using Global Settings")
                start_date = global_start
                end_date = global_end
                sma_windows = global_sma_windows
                st.text(f"Start: {start_date}")
                st.text(f"End: {end_date}")
                st.text(f"SMA: {', '.join(map(str, sma_windows))}")
            else:
                start_date = st.date_input(f"{t} Start Date", value=min_date, min_value=min_date, max_value=max_date, key=f"start_{t}")
                end_date = st.date_input(f"{t} End Date", value=max_date, min_value=min_date, max_value=max_date, key=f"end_{t}")
                if start_date > end_date:
                    st.error(f"Start Date must be before End Date for {t}")
                    st.stop()
                # Wait for user input, no default SMA
                sma_input = st.text_input(f"{t} SMA Windows (comma-separated)", value="", key=f"sma_{t}")
                if not sma_input:
                    st.info(f"Enter SMA window(s) for {t} to continue")
                    st.stop()
                sma_windows = [int(x.strip()) for x in sma_input.split(",") if x.strip().isdigit() and int(x) > 0]
                if not sma_windows:
                    st.error(f"Invalid SMA windows for {t}. Please enter positive integers.")
                    st.stop()
            
            ticker_settings[t] = {"start_date": start_date, "end_date": end_date, "sma_windows": sma_windows}

# Load Data: yfinance
else:
    tickers_input = st.sidebar.text_input("Enter tickers (comma-separated)")
    if not tickers_input:
        st.info("Please enter at least one ticker to continue.")
        st.stop()
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    st.sidebar.markdown("### Per-Ticker Settings")
    for t in tickers:
        hist = yf.Ticker(t).history(period="max")
        if hist.empty:
            st.warning(f"No data available for {t}, skipping.")
            continue
        min_date, max_date = hist.index.min().date(), hist.index.max().date()
        
        with st.sidebar.expander(f"📊 {t}", expanded=not use_global):
            if use_global:
                st.info("🌐 Using Global Settings")
                start_date = global_start
                end_date = global_end
                sma_windows = global_sma_windows
                st.text(f"Start: {start_date}")
                st.text(f"End: {end_date}")
                st.text(f"SMA: {', '.join(map(str, sma_windows))}")
            else:
                start_date = st.date_input(f"{t} Start Date", value=max_date - datetime.timedelta(days=90),
                                           min_value=min_date, max_value=max_date, key=f"start_{t}")
                end_date = st.date_input(f"{t} End Date", value=max_date, min_value=min_date, max_value=max_date, key=f"end_{t}")
                if start_date > end_date:
                    st.error(f"Start Date must be before End Date for {t}")
                    st.stop()
                # Wait for user input, no default SMA
                sma_input = st.text_input(f"{t} SMA Windows (comma-separated)", value="", key=f"sma_{t}")
                if not sma_input:
                    st.info(f"Enter SMA window(s) for {t} to continue")
                    st.stop()
                sma_windows = [int(x.strip()) for x in sma_input.split(",") if x.strip().isdigit() and int(x) > 0]
                if not sma_windows:
                    st.error(f"Invalid SMA windows for {t}. Please enter positive integers.")
                    st.stop()
            
            ticker_settings[t] = {"start_date": start_date, "end_date": end_date, "sma_windows": sma_windows}

    # Fetch & process each ticker individually
    all_cleaned = pd.DataFrame()
    for t, settings in ticker_settings.items():
        cleaned_raw = fetch_stock_data([t], str(settings["start_date"]), str(settings["end_date"]))
        if cleaned_raw.empty:
            st.warning(f"No data returned for {t}")
            continue
        cleaned_t = clean_all_tickers(cleaned_raw, [t])
        cleaned_t = align_with_trading_days(cleaned_t, str(settings["start_date"]), str(settings["end_date"]))
        cleaned_t = detect_outliers(cleaned_t)
        validate_schema(cleaned_t)
        validate_preprocessing(cleaned_t, str(settings["start_date"]), str(settings["end_date"]))

        # Apply per-ticker SMA windows
        cleaned_t = calculate_sma(cleaned_t, settings["sma_windows"])
        all_cleaned = pd.concat([all_cleaned, cleaned_t])

    cleaned = all_cleaned.copy()
    if cleaned.empty:
        st.error("No valid data available after processing all tickers.")
        st.stop()

# Analytics
analytics_data = calculate_daily_returns(cleaned)
analytics_data = detect_streaks(analytics_data)

# Determine common period for fair comparison
if ticker_settings:
    start_dates = [s["start_date"] for s in ticker_settings.values()]
    end_dates = [s["end_date"] for s in ticker_settings.values()]
    common_start = max(start_dates)
    common_end = min(end_dates)

    if common_start > common_end:
        st.warning("Tickers have no overlapping date range. Risk-return and profit comparison may not be accurate.")
        comparison_data = pd.DataFrame()
    else:
        comparison_data = analytics_data[
            (analytics_data["Date"] >= pd.Timestamp(common_start)) &
            (analytics_data["Date"] <= pd.Timestamp(common_end))
        ]
else:
    comparison_data = pd.DataFrame()

# Calculate risk-return and profit summary only for the comparison period
if not comparison_data.empty:
    risk_return_summary = calculate_annual_risk_return(comparison_data, risk_free_rate=0.03)
    profit_summary = calculate_profits(comparison_data, tickers)
else:
    risk_return_summary = pd.DataFrame()
    profit_summary = pd.DataFrame()

# Tabs
tabs = st.tabs(["Data Quality", "Analytics & Streaks", "Risk-Return", "Profits"])

# Tab 1: Data Quality
with tabs[0]:
    st.subheader("📝 Data Quality Report")
    
    # View Data Quality Table (Outlier column removed for front-end display)
    with st.expander("View Data Quality Table", expanded=True):
        dq_report = data_quality_report(cleaned).copy()
        # Remove the Outlier column for display only
        dq_report_display = dq_report.drop(columns=["Outliers (Z%)"], errors="ignore")
        st.dataframe(format_df(dq_report_display))
    
    # Download Data / Charts (full data including Outlier)
    with st.expander("Download Data/Charts", expanded=True):
        st.download_button(
            "💾 Download Cleaned Data",
            cleaned.to_csv(index=False, float_format="%.2f").encode(),
            "cleaned_data.csv"
        )


# Tab 2: Analytics & Streaks
with tabs[1]:
    st.subheader("📈 Price & Streak Charts")
    if not tickers:
        st.warning("No tickers available for chart plotting.")
    else:
        selected_ticker = st.selectbox("Select Ticker for Charts", tickers, index=0)
        if selected_ticker:
            subset = analytics_data[analytics_data["Ticker"] == selected_ticker].sort_values("Date")
            
            # Price + SMA section
            with st.expander(f"{selected_ticker} — Price + SMA Chart", expanded=True):
                col1, col2, col3 = st.columns(3)
                col1.metric("High Price", f"${subset['Close'].max():.2f}")
                col2.metric("Low Price", f"${subset['Close'].min():.2f}")
                total_return = (subset['Close'].iloc[-1] / subset['Close'].iloc[0] - 1) * 100
                col3.metric("Total Return", f"{total_return:.2f}%")
                
                sma_windows = ticker_settings.get(selected_ticker, {}).get("sma_windows", [])
                fig_price_sma = plot_price_sma(analytics_data, selected_ticker, sma_windows)
                
                # Download CSV first
                st.download_button(
                    f"💾 Download Analytics Data (CSV)",
                    subset.to_csv(index=False, float_format="%.2f").encode(),
                    file_name=f"{selected_ticker}_analytics.csv"
                )
                
                # Then download chart
                if fig_price_sma:
                    st.download_button(
                        f"💾 Download Price + SMA Chart (HTML)",
                        data=fig_price_sma.to_html(include_plotlyjs='cdn'),
                        file_name=f"{selected_ticker}_price_sma.html"
                    )

            # Streak section
            with st.expander(f"{selected_ticker} — Streak Chart", expanded=True):
                longest_up = subset["Streak"].loc[subset["Streak"] > 0].max() or 0
                longest_down = subset["Streak"].loc[subset["Streak"] < 0].min() or 0
                pos_days = (subset["Daily_Return"] > 0).sum()
                neg_days = (subset["Daily_Return"] < 0).sum()
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Longest Up Streak", longest_up)
                col2.metric("Longest Down Streak", longest_down)
                col3.metric("Positive Days", pos_days)
                col4.metric("Negative Days", neg_days)
                
                fig_streaks = plot_streaks(analytics_data, selected_ticker)
                
                # Download streak summary CSV first
                streak_summary = pd.DataFrame([{
                    "Ticker": selected_ticker,
                    "Longest_Up_Streak": longest_up,
                    "Longest_Down_Streak": longest_down,
                    "Total_Up_Days": pos_days,
                    "Total_Down_Days": neg_days
                }])
                st.download_button(
                    f"💾 Download Streak Summary (CSV)",
                    streak_summary.to_csv(index=False, float_format="%.2f").encode(),
                    file_name=f"{selected_ticker}_streak_summary.csv"
                )
                
                # Then download streak chart
                if fig_streaks:
                    st.download_button(
                        f"💾 Download Streak Chart (HTML)",
                        data=fig_streaks.to_html(include_plotlyjs='cdn'),
                        file_name=f"{selected_ticker}_streaks.html"
                    )


# Tab 3: Risk-Return (Refactored with metrics & unique keys)
with tabs[2]:
    st.subheader("📊 Annual Risk vs Return (Sharpe Highlighted)")
    if not risk_return_summary.empty:
        selected_rr_tickers = st.multiselect(
            "Select Ticker(s) for Risk-Return Chart", tickers, default=tickers
        )
        rr_chart_df = risk_return_summary[risk_return_summary["Ticker"].isin(selected_rr_tickers)]
        with st.expander("View Metrics & Risk-Return Chart", expanded=True):
            if not rr_chart_df.empty:
                # -------------------------
                # Metrics at the top
                # -------------------------
                for t, row in rr_chart_df.iterrows():
                    col1, col2, col3 = st.columns(3)
                    col1.metric(f"{row['Ticker']} — Annual Return", f"{row['Annual_Return']:.2%}")
                    col2.metric(f"{row['Ticker']} — Volatility", f"{row['Annual_Volatility']:.2%}")
                    col3.metric(f"{row['Ticker']} — Sharpe Ratio", f"{row['Sharpe_Ratio']:.2f}")
                
                # -------------------------
                # Plot Risk-Return Chart
                # -------------------------
                fig_risk_return = plot_annual_risk_return(rr_chart_df)
                
                # -------------------------
                # Download Buttons
                # -------------------------
                st.download_button(
                    "💾 Download Risk-Return Summary (CSV)",
                    rr_chart_df.to_csv(index=False, float_format="%.4f").encode(),
                    "risk_return_summary.csv"
                )
                if fig_risk_return:
                    st.download_button(
                        "💾 Download Risk-Return Chart (HTML)",
                        data=fig_risk_return.to_html(include_plotlyjs='cdn'),
                        file_name="annual_risk_return.html"
                    )
            else:
                st.warning("No risk-return data available for selected tickers.")
    else:
        st.warning("Risk-return summary is empty.")


# Tab 4: Profits
with tabs[3]:
    st.subheader("📈 Best Buy/Sell Opportunities")
    if not tickers:
        st.warning("No tickers available for buy/sell plotting.")
    else:
        selected_profit_ticker = st.selectbox(
            "Select Ticker to View Best Buy/Sell Points", 
            tickers, 
            key="buy_sell_selector"
        )
        if selected_profit_ticker:
            with st.expander(f"{selected_profit_ticker} — Buy/Sell Chart", expanded=True):
                # Display metrics at the top
                ticker_profit = profit_summary[profit_summary["Ticker"] == selected_profit_ticker]
                if not ticker_profit.empty:
                    row = ticker_profit.iloc[0]
                    col1, col2, col3, col4 = st.columns(4)
                    
                    # Format dates as strings
                    buy_date_str = pd.to_datetime(row['Best_Buy_Date']).strftime('%Y-%m-%d') if pd.notna(row['Best_Buy_Date']) else "N/A"
                    sell_date_str = pd.to_datetime(row['Best_Sell_Date']).strftime('%Y-%m-%d') if pd.notna(row['Best_Sell_Date']) else "N/A"
                    
                    col1.metric("Best Buy Date", buy_date_str)
                    col2.metric("Best Sell Date", sell_date_str)
                    col3.metric("Return %", f"{row['Return_Pct']:.2f}%")
                
                # Plot chart
                fig_buy_sell = plot_best_buy_sell(analytics_data, selected_profit_ticker)
                if fig_buy_sell:
                    st.download_button(
                        f"💾 {selected_profit_ticker} Buy/Sell Chart (HTML)",
                        data=fig_buy_sell.to_html(include_plotlyjs='cdn'),
                        file_name=f"{selected_profit_ticker}_buy_sell.html"
                    )
    
    st.subheader("💰 Profit Comparison")
    with st.expander("View Profit Metrics & Chart", expanded=True):
        if not profit_summary.empty:
            # Metrics at the top
            for idx, row in profit_summary.iterrows():
                col1, col2 = st.columns(2)
                col1.metric(f"{row['Ticker']} — Max Profit (Single)", f"${row['MaxProfit_Single']:.2f}")
                col2.metric(f"{row['Ticker']} — Max Profit (Multiple)", f"${row['MaxProfit_Multiple']:.2f}")
            
            # Profit Comparison Chart
            fig_profit = plot_profit_comparison(profit_summary)
            
            # Download Buttons
            st.download_button(
                "💾 Download Profit Summary (CSV)",
                profit_summary.to_csv(index=False, float_format="%.2f").encode(),
                "profit_summary.csv"
            )
            if fig_profit:
                st.download_button(
                    "💾 Download Profit Comparison Chart (HTML)",
                    data=fig_profit.to_html(include_plotlyjs='cdn'),
                    file_name="profit_comparison.html"
                )
        else:
            st.warning("No profit data available.")