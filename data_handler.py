# data_handler.py

import os
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_market_calendars as mcal
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================
# Display & Plot Settings
# ==========================
pd.set_option("display.max_columns", None)
plt.style.use("seaborn-v0_8")

# ==========================
# Configuration
# ==========================
CONFIG = {
    "tickers": ["AAPL", "MSFT", "GOOGL", "JPM", "TSLA"],
    "start_date": "2022-01-01",
    "end_date": "2025-01-01",
    "backup_path": "data/backup_stocks.csv",
    "output_long": "data/cleaned_stock_data_long.csv",
    "output_wide": "data/cleaned_stock_data_wide.csv",
}


# ==========================
# Data Acquisition
# ==========================
def fetch_stock_data(tickers: list[str], start: str, end: str, backup_path: str) -> pd.DataFrame:
    """Fetch historical stock data from Yahoo Finance with CSV fallback."""
    try:
        if not tickers:
            raise ValueError("No tickers provided")
        data = yf.download(
            tickers, start=start, end=end, group_by="ticker", auto_adjust=True, progress=False
        )
        if data.empty:
            raise ValueError("No data fetched; check tickers or dates")
        print(f"✅ Data fetched successfully for {len(tickers)} tickers ({start} → {end})")
        return data
    except Exception as e:
        print(f"⚠️ yfinance failed: {e}. Loading backup CSV instead...")
        return pd.read_csv(backup_path)


# ==========================
# Data Cleaning
# ==========================
def clean_stock_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Clean a single ticker's stock data."""
    df = df.copy().reset_index()
    df["Ticker"] = ticker

    # Validate
    if df.empty or "Date" not in df:
        raise ValueError(f"❌ No data or missing Date column for {ticker}")
    if (df[["Open", "High", "Low", "Close"]] < 0).any().any():
        raise ValueError(f"❌ Negative prices detected for {ticker}")

    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df.sort_values("Date", inplace=True)

    # Missing values
    missing_pct = df[["Open", "High", "Low", "Close", "Volume"]].isnull().mean(axis=1)
    df = df[missing_pct < 0.5].ffill().bfill()
    
    df.drop_duplicates(subset=["Date"], inplace=True)
    
    # Numeric types
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    return df


def clean_all_tickers(raw_data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Apply cleaning to all tickers in the dataset."""
    cleaned_list = []
    for ticker in tickers:
        temp = raw_data[ticker].reset_index()
        cleaned_list.append(clean_stock_data(temp, ticker))
        print(f"✅ {ticker} cleaned successfully")
    cleaned_data = pd.concat(cleaned_list, ignore_index=True)
    if "index" in cleaned_data.columns:
        cleaned_data.drop(columns=["index"], inplace=True)
    return cleaned_data


# ==========================
# Trading Day Alignment
# ==========================
def align_with_trading_days(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Filter data to official NYSE trading days."""
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=start, end_date=end)
    trading_days = schedule.index.normalize()

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df = df[df["Date"].isin(trading_days)]
    print(f"✅ Data aligned with NYSE trading days ({trading_days.min().date()} → {trading_days.max().date()})")
    return df


# ==========================
# Outlier Detection
# ==========================
def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Add daily returns and detect outliers via Z-score and IQR."""
    df = df.copy()
    df["Daily_Return"] = df.groupby("Ticker")["Close"].pct_change()
    df["Z_Score"] = df.groupby("Ticker")["Daily_Return"].transform(lambda x: stats.zscore(x, nan_policy="omit"))
    df["Outlier_Z"] = df["Z_Score"].abs() > 3

    def iqr_outliers(series):
        q1, q3 = np.nanpercentile(series, [25, 75])
        iqr = q3 - q1
        return (series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))
    
    df["Outlier_IQR"] = df.groupby("Ticker")["Daily_Return"].transform(iqr_outliers)
    print("✅ Outlier detection completed")
    return df


# # ==========================
# # Plotting Functions
# # ==========================
# def plot_outliers(df: pd.DataFrame, tickers: list[str] | str):
#     """Plot daily returns with outliers highlighted."""
#     if isinstance(tickers, str):
#         tickers = [tickers]

#     for ticker in tickers:
#         subset = df[df["Ticker"] == ticker]
#         plt.figure(figsize=(12, 6))
#         plt.plot(subset["Date"], subset["Daily_Return"], label="Daily Returns", alpha=0.6)
#         plt.scatter(subset["Date"][subset["Outlier_Z"]],
#                     subset["Daily_Return"][subset["Outlier_Z"]],
#                     color="red", label="Z-Score Outliers")
#         plt.scatter(subset["Date"][subset["Outlier_IQR"]],
#                     subset["Daily_Return"][subset["Outlier_IQR"]],
#                     color="orange", marker="x", label="IQR Outliers")
#         plt.title(f"Outlier Detection in {ticker} Daily Returns")
#         plt.xlabel("Date")
#         plt.ylabel("Daily Return")
#         plt.legend()
#         plt.show()


# ==========================
# Data Quality & Validation
# ==========================
def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a data quality summary per ticker."""
    core_cols = ["Open", "High", "Low", "Close", "Volume"]
    report = []
    nyse = mcal.get_calendar("NYSE")
    for ticker, group in df.groupby("Ticker"):
        total_rows = len(group)
        missing_pct = (group[core_cols].isnull().sum().sum() / (total_rows * len(core_cols))) * 100
        duplicates = group.duplicated(subset=["Date"]).sum()
        min_date, max_date = group["Date"].min(), group["Date"].max()
        outlier_pct = group["Outlier_Z"].mean() * 100 if "Outlier_Z" in group else np.nan
        schedule = nyse.schedule(start_date=min_date, end_date=max_date)
        expected_days = len(schedule)
        coverage_pct = (total_rows / expected_days) * 100 if expected_days > 0 else np.nan

        report.append({
            "Ticker": ticker,
            "Rows": total_rows,
            "Date Range": f"{min_date.date()} → {max_date.date()}",
            "Missing %": f"{missing_pct:.2f}%",
            "Duplicates": duplicates,
            "Outliers (Z%)": f"{outlier_pct:.2f}%" if not np.isnan(outlier_pct) else "N/A",
            "Trading Days Coverage %": f"{coverage_pct:.2f}%"
        })
    return pd.DataFrame(report)


def validate_preprocessing(df: pd.DataFrame, start_date: str, end_date: str):
    """Automated checks for preprocessing correctness."""
    assert df[["Open", "High", "Low", "Close", "Volume"]].isnull().sum().sum() == 0, "❌ Missing values remain!"
    for ticker, group in df.groupby("Ticker"):
        expected_returns = group["Close"].pct_change()
        assert np.allclose(group["Daily_Return"].dropna(), expected_returns.dropna(), equal_nan=True), \
            f"❌ Daily returns mismatch for {ticker}"
        assert (group[["Open", "High", "Low", "Close"]] >= 0).all().all(), f"❌ Negative prices detected for {ticker}"
        assert not group["Date"].duplicated().any(), f"❌ Duplicates found in {ticker}"
        assert group["Date"].min() >= pd.to_datetime(start_date), f"❌ Start date invalid for {ticker}"
        assert group["Date"].max() <= pd.to_datetime(end_date), f"❌ End date invalid for {ticker}"
    print("✅ Preprocessing validations passed!")


# ==========================
# Wide Format Transformation
# ==========================
def pivot_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-format cleaned data into wide format (Close prices)."""
    pivot_data = df.pivot(index="Date", columns="Ticker", values="Close").ffill().bfill()
    return pivot_data


# ==========================
# Save Datasets
# ==========================
def save_datasets(cleaned_data: pd.DataFrame, pivot_data: pd.DataFrame, config: dict):
    """Save long and wide format datasets to disk."""
    os.makedirs(os.path.dirname(config["output_long"]), exist_ok=True)
    os.makedirs(os.path.dirname(config["output_wide"]), exist_ok=True)

    cleaned_data.to_csv(config["output_long"], index=False)
    pivot_data.to_csv(config["output_wide"])
    print(f"✅ Datasets saved: {config['output_long']}, {config['output_wide']}")
