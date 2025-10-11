"""
main.py

Streamlit dashboard for Stock Market Trend Analysis:
- Load data from CSV or Yahoo Finance
- Clean, validate, and process stock data
- Calculate daily returns, SMA, streaks, risk-return, and profits
- Visualize trends, charts, and summary metrics with delta pills
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

def calculate_period_change(subset):
    """Calculate price change over the period"""
    if len(subset) < 2:
        return 0
    start_price = subset['Close'].iloc[0]
    end_price = subset['Close'].iloc[-1]
    return ((end_price - start_price) / start_price) * 100

def calculate_ytd_change(subset):
    """Calculate YTD change if data includes current year"""
    current_year = datetime.date.today().year
    ytd_data = subset[subset['Date'].dt.year == current_year]
    if len(ytd_data) < 2:
        return None
    start_price = ytd_data['Close'].iloc[0]
    end_price = ytd_data['Close'].iloc[-1]
    return ((end_price - start_price) / start_price) * 100

def calculate_volatility_change(subset):
    """Calculate change in volatility (comparing first half vs second half)"""
    if len(subset) < 20:
        return None
    mid_point = len(subset) // 2
    first_half_vol = subset['Daily_Return'].iloc[:mid_point].std()
    second_half_vol = subset['Daily_Return'].iloc[mid_point:].std()
    if first_half_vol == 0:
        return None
    return ((second_half_vol - first_half_vol) / first_half_vol) * 100


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

# Calculate individual ticker metrics (always available)
individual_risk_return = pd.DataFrame()
individual_profit = pd.DataFrame()

for ticker in tickers:
    ticker_data = analytics_data[analytics_data["Ticker"] == ticker]
    if not ticker_data.empty:
        # Calculate risk-return for individual ticker
        ticker_rr = calculate_annual_risk_return(ticker_data, risk_free_rate=0.03)
        individual_risk_return = pd.concat([individual_risk_return, ticker_rr])
        
        # Calculate profit for individual ticker
        ticker_profit = calculate_profits(ticker_data, [ticker])
        individual_profit = pd.concat([individual_profit, ticker_profit])

# Determine common period for fair comparison (optional, for multi-ticker comparison)
comparison_data = pd.DataFrame()
risk_return_summary = pd.DataFrame()
profit_summary = pd.DataFrame()

if ticker_settings and len(tickers) > 1:
    start_dates = [s["start_date"] for s in ticker_settings.values()]
    end_dates = [s["end_date"] for s in ticker_settings.values()]
    common_start = max(start_dates)
    common_end = min(end_dates)

    if common_start <= common_end:
        comparison_data = analytics_data[
            (analytics_data["Date"] >= pd.Timestamp(common_start)) &
            (analytics_data["Date"] <= pd.Timestamp(common_end))
        ]
        
        # Calculate risk-return and profit summary for comparison period
        if not comparison_data.empty:
            risk_return_summary = calculate_annual_risk_return(comparison_data, risk_free_rate=0.03)
            profit_summary = calculate_profits(comparison_data, tickers)

# Tabs
tabs = st.tabs(["📊 Overview", "📝 Data Quality", "📈 Analytics & Streaks", "⚖️ Risk-Return", "💰 Profits"])

# Tab 0: Overview Dashboard
with tabs[0]:
    st.subheader("📊 Portfolio Overview")
    
    # Summary metrics across all tickers
    if not analytics_data.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        # Portfolio-wide metrics
        total_tickers = len(tickers)
        avg_return = analytics_data.groupby('Ticker')['Daily_Return'].mean().mean() * 252 * 100
        avg_volatility = analytics_data.groupby('Ticker')['Daily_Return'].std().mean() * (252**0.5) * 100
        
        # Calculate period returns for each ticker
        period_returns = []
        for t in tickers:
            t_data = analytics_data[analytics_data['Ticker'] == t].sort_values('Date')
            if len(t_data) >= 2:
                period_ret = calculate_period_change(t_data)
                period_returns.append(period_ret)
        
        avg_period_return = sum(period_returns) / len(period_returns) if period_returns else 0
        
        col1.metric("Total Tickers", total_tickers)
        col2.metric("Avg. Annual Return", f"{avg_return:.2f}%", 
                   delta=f"{avg_period_return:.2f}% Period")
        col3.metric("Avg. Volatility", f"{avg_volatility:.2f}%")
        col4.metric("Trading Days", len(analytics_data['Date'].unique()))
    
    # Individual ticker summary cards
    st.markdown("### Ticker Performance Summary")
    
    # Create cards for each ticker
    for ticker in tickers:
        ticker_data = analytics_data[analytics_data['Ticker'] == ticker].sort_values('Date')
        if ticker_data.empty:
            continue
            
        with st.expander(f"📈 {ticker} - Quick Stats", expanded=True):
            col1, col2, col3, col4, col5 = st.columns(5)
            
            # Current/Latest price
            latest_price = ticker_data['Close'].iloc[-1]
            period_change = calculate_period_change(ticker_data)
            
            col1.metric("Latest Price", f"${latest_price:.2f}", 
                       delta=f"{period_change:.2f}%")
            
            # High/Low with delta from current
            high_price = ticker_data['Close'].max()
            low_price = ticker_data['Close'].min()
            high_delta = ((latest_price - high_price) / high_price * 100)
            low_delta = ((latest_price - low_price) / low_price * 100)
            
            col2.metric("Period High", f"${high_price:.2f}", 
                       delta=f"{high_delta:.2f}%" if high_delta != 0 else None)
            col3.metric("Period Low", f"${low_price:.2f}", 
                       delta=f"{low_delta:.2f}%" if low_delta != 0 else None)
            
            # Volatility metrics
            current_vol = ticker_data['Daily_Return'].std() * (252**0.5) * 100
            vol_change = calculate_volatility_change(ticker_data)
            
            col4.metric("Volatility (Annual)", f"{current_vol:.2f}%",
                       delta=f"{vol_change:.1f}%" if vol_change else None,
                       delta_color="inverse")  # Higher volatility is typically bad
            
            # Win rate
            positive_days = (ticker_data['Daily_Return'] > 0).sum()
            total_days = len(ticker_data[ticker_data['Daily_Return'].notna()])
            win_rate = (positive_days / total_days * 100) if total_days > 0 else 0
            
            col5.metric("Win Rate", f"{win_rate:.1f}%",
                       delta=f"{positive_days}/{total_days} days")

# Tab 1: Data Quality
with tabs[1]:
    st.subheader("📝 Data Quality Report")
    
    # Summary metrics for data quality
    col1, col2, col3 = st.columns(3)
    total_records = len(cleaned)
    complete_tickers = cleaned['Ticker'].nunique()
    date_range = (cleaned['Date'].max() - cleaned['Date'].min()).days
    
    col1.metric("Total Records", f"{total_records:,}")
    col2.metric("Complete Tickers", complete_tickers)
    col3.metric("Date Range (Days)", date_range)
    
    # View Data Quality Table (Outlier column removed for front-end display)
    with st.expander("View Data Quality Table", expanded=True):
        dq_report = data_quality_report(cleaned).copy()
        # Remove the Outlier column for display only
        dq_report_display = dq_report.drop(columns=["Outliers (Z%)"], errors="ignore")
        st.dataframe(format_df(dq_report_display), use_container_width=True)
    
    # Download Data (full data including Outlier)
    with st.expander("Download Data", expanded=True):
        st.download_button(
            "💾 Download Cleaned Data",
            cleaned.to_csv(index=False, float_format="%.2f").encode(),
            "cleaned_data.csv"
        )

# Tab 2: Analytics & Streaks
with tabs[2]:
    st.subheader("📈 Price & Streak Charts")
    if not tickers:
        st.warning("No tickers available for chart plotting.")
    else:
        selected_ticker = st.selectbox("Select Ticker for Charts", tickers, index=0)
        if selected_ticker:
            subset = analytics_data[analytics_data["Ticker"] == selected_ticker].sort_values("Date")
            
            # Price + SMA section
            with st.expander(f"{selected_ticker} — Price + SMA Chart", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                
                high_price = subset['Close'].max()
                low_price = subset['Close'].min()
                current_price = subset['Close'].iloc[-1]
                total_return = calculate_period_change(subset)
                
                # Calculate distance from high/low
                from_high = ((current_price - high_price) / high_price * 100)
                from_low = ((current_price - low_price) / low_price * 100)
                
                col1.metric("High Price", f"${high_price:.2f}",
                           delta=f"{from_high:.1f}% from high")
                col2.metric("Low Price", f"${low_price:.2f}",
                           delta=f"{from_low:.1f}% from low")
                col3.metric("Total Return", f"{total_return:.2f}%",
                           delta="Period Performance")
                
                # YTD Return if applicable
                ytd_return = calculate_ytd_change(subset)
                if ytd_return is not None:
                    col4.metric("YTD Return", f"{ytd_return:.2f}%",
                               delta="Year to Date")
                else:
                    col4.metric("Current Price", f"${current_price:.2f}")
                
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
                longest_down = abs(subset["Streak"].loc[subset["Streak"] < 0].min() or 0)
                pos_days = (subset["Daily_Return"] > 0).sum()
                neg_days = (subset["Daily_Return"] < 0).sum()
                total_days = pos_days + neg_days
                win_rate = (pos_days / total_days * 100) if total_days > 0 else 0
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Longest Up Streak", f"{longest_up} days",
                           delta="Consecutive gains")
                col2.metric("Longest Down Streak", f"{longest_down} days",
                           delta="Consecutive losses",
                           delta_color="inverse")
                col3.metric("Positive Days", pos_days,
                           delta=f"{win_rate:.1f}% win rate")
                col4.metric("Negative Days", neg_days,
                           delta=f"{100-win_rate:.1f}% loss rate",
                           delta_color="inverse")
                
                fig_streaks = plot_streaks(analytics_data, selected_ticker)
                
                # Download streak summary CSV first
                streak_summary = pd.DataFrame([{
                    "Ticker": selected_ticker,
                    "Longest_Up_Streak": longest_up,
                    "Longest_Down_Streak": longest_down,
                    "Total_Up_Days": pos_days,
                    "Total_Down_Days": neg_days,
                    "Win_Rate": win_rate
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

# Tab 3: Risk-Return
with tabs[3]:
    st.subheader("⚖️ Annual Risk vs Return (Sharpe Highlighted)")
    
    # Always show individual ticker data
    if not individual_risk_return.empty:
        selected_rr_tickers = st.multiselect(
            "Select Ticker(s) for Risk-Return Analysis", 
            tickers, 
            default=tickers,
            key="risk_return_selector"
        )
        
        if selected_rr_tickers:
            # Get data for selected tickers from individual analysis
            rr_chart_df = individual_risk_return[individual_risk_return["Ticker"].isin(selected_rr_tickers)].copy()
            # Reset index so the plot function can properly identify best Sharpe
            rr_chart_df = rr_chart_df.reset_index(drop=True)
            
            with st.expander("View Risk-Return Metrics & Chart", expanded=True):
                if not rr_chart_df.empty:
                    # Metrics for each selected ticker
                    for idx, row in rr_chart_df.iterrows():
                        col1, col2, col3 = st.columns(3)
                        
                        # Calculate risk-free rate excess for context
                        excess_return = row['Annual_Return'] - 0.03  # Risk-free rate is 3%
                        
                        col1.metric(f"{row['Ticker']} — Annual Return", 
                                   f"{row['Annual_Return']:.2%}",
                                   delta=f"{excess_return:.2%} excess")
                        col2.metric(f"{row['Ticker']} — Volatility", 
                                   f"{row['Annual_Volatility']:.2%}",
                                   delta=f"Risk level",
                                   delta_color="off")
                        col3.metric(f"{row['Ticker']} — Sharpe Ratio", 
                                   f"{row['Sharpe_Ratio']:.2f}",
                                   delta="Risk-adjusted" if row['Sharpe_Ratio'] > 1 else "Below 1.0",
                                   delta_color="normal" if row['Sharpe_Ratio'] > 1 else "inverse")
                    
                    # Plot Risk-Return Chart
                    fig_risk_return = plot_annual_risk_return(rr_chart_df)
                    
                    # Download Buttons
                    csv_filename = f"{selected_rr_tickers[0]}_risk_return.csv" if len(selected_rr_tickers) == 1 else "risk_return_comparison.csv"
                    html_filename = f"{selected_rr_tickers[0]}_risk_return.html" if len(selected_rr_tickers) == 1 else "risk_return_comparison.html"
                    
                    st.download_button(
                        "💾 Download Risk-Return Data (CSV)",
                        rr_chart_df.to_csv(index=False, float_format="%.4f").encode(),
                        csv_filename
                    )
                    if fig_risk_return:
                        st.download_button(
                            "💾 Download Risk-Return Chart (HTML)",
                            data=fig_risk_return.to_html(include_plotlyjs='cdn'),
                            file_name=html_filename
                        )
                else:
                    st.warning("No risk-return data available for selected tickers.")
        else:
            st.info("Please select at least one ticker to view risk-return analysis.")
    else:
        st.warning("No risk-return data available.")

# Tab 4: Profits
with tabs[4]:
    st.subheader("💰 Best Buy/Sell Opportunities")
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
                # Display metrics at the top using individual data
                ticker_profit = individual_profit[individual_profit["Ticker"] == selected_profit_ticker]
                    
                if not ticker_profit.empty:
                    row = ticker_profit.iloc[0]
                    col1, col2, col3, col4 = st.columns(4)
                    
                    # Format dates as strings
                    buy_date_str = pd.to_datetime(row['Best_Buy_Date']).strftime('%Y-%m-%d') if pd.notna(row['Best_Buy_Date']) else "N/A"
                    sell_date_str = pd.to_datetime(row['Best_Sell_Date']).strftime('%Y-%m-%d') if pd.notna(row['Best_Sell_Date']) else "N/A"
                    
                    col1.metric("Best Buy Date", buy_date_str)
                    col2.metric("Best Sell Date", sell_date_str)
                    col3.metric("Return %", f"{row['Return_Pct']:.2f}%",
                               delta="Single trade opportunity")
                    
                    # Add holding period
                    if pd.notna(row['Best_Buy_Date']) and pd.notna(row['Best_Sell_Date']):
                        holding_days = (pd.to_datetime(row['Best_Sell_Date']) - pd.to_datetime(row['Best_Buy_Date'])).days
                        col4.metric("Holding Period", f"{holding_days} days")
                else:
                    st.warning(f"No profit data available for {selected_profit_ticker}.")
                
                # Plot chart
                fig_buy_sell = plot_best_buy_sell(analytics_data, selected_profit_ticker)
                if fig_buy_sell:
                    st.download_button(
                        f"💾 {selected_profit_ticker} Buy/Sell Chart (HTML)",
                        data=fig_buy_sell.to_html(include_plotlyjs='cdn'),
                        file_name=f"{selected_profit_ticker}_buy_sell.html"
                    )
    
    st.markdown("---")
    st.subheader("💰 Profit Summary")
    
    # Always show individual profit data
    if not individual_profit.empty:
        selected_profit_tickers = st.multiselect(
            "Select Ticker(s) for Profit Analysis", 
            tickers, 
            default=tickers,
            key="profit_summary_selector"
        )
        
        if selected_profit_tickers:
            profit_chart_df = individual_profit[individual_profit["Ticker"].isin(selected_profit_tickers)]
            
            with st.expander("View Profit Metrics & Chart", expanded=True):
                if not profit_chart_df.empty:
                    # Metrics for each selected ticker
                    for idx, row in profit_chart_df.iterrows():
                        col1, col2 = st.columns(2)
                        
                        # Calculate profit multiplier (multiple vs single)
                        if row['MaxProfit_Single'] > 0:
                            profit_multiplier = row['MaxProfit_Multiple'] / row['MaxProfit_Single']
                            multiplier_text = f"{profit_multiplier:.2f}x multiple"
                        else:
                            multiplier_text = "N/A"
                        
                        col1.metric(f"{row['Ticker']} — Max Profit (Single)", 
                                   f"${row['MaxProfit_Single']:.2f}",
                                   delta="Best single trade")
                        col2.metric(f"{row['Ticker']} — Max Profit (Multiple)", 
                                   f"${row['MaxProfit_Multiple']:.2f}",
                                   delta=multiplier_text)
                    
                    # Profit Comparison Chart
                    fig_profit = plot_profit_comparison(profit_chart_df)
                    
                    # Download Buttons
                    csv_filename = f"{selected_profit_tickers[0]}_profit.csv" if len(selected_profit_tickers) == 1 else "profit_comparison.csv"
                    html_filename = f"{selected_profit_tickers[0]}_profit.html" if len(selected_profit_tickers) == 1 else "profit_comparison.html"
                    
                    st.download_button(
                        "💾 Download Profit Data (CSV)",
                        profit_chart_df.to_csv(index=False, float_format="%.2f").encode(),
                        csv_filename
                    )
                    if fig_profit:
                        st.download_button(
                            "💾 Download Profit Chart (HTML)",
                            data=fig_profit.to_html(include_plotlyjs='cdn'),
                            file_name=html_filename
                        )
                else:
                    st.warning("No profit data available for selected tickers.")
        else:
            st.info("Please select at least one ticker to view profit analysis.")
    else:
        st.warning("No profit data available.")