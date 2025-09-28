# data_handler.py

import os
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_market_calendars as mcal
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from config import CONFIG

# Display & Plotting settings
pd.set_option("display.max_columns", None)
plt.style.use("seaborn-v0_8")

print("Packages imported. CONFIG loaded.")


# 1) Data Acquisition
def fetch_stock_data(tickers: list[str], start: str, end: str, backup_path: str) -> pd.DataFrame:
    """Fetch historical stock data from Yahoo Finance with CSV fallback."""
    if not tickers:
        raise ValueError("No tickers provided")

    try:
        data = yf.download(
            tickers,
            start=start,
            end=end,
            group_by="ticker",
            auto_adjust=True,
            progress=False
        )
        if data.empty:
            raise ValueError("No data fetched; check tickers or dates")
        print(f"Data fetched successfully for {len(tickers)} tickers ({start} → {end})")
        return data
    except Exception as e:
        print(f"yfinance failed: {e}. Loading backup CSV instead...")
        return pd.read_csv(backup_path, parse_dates=["Date"])


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame, parse 'Date' column."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    df = pd.read_csv(path, parse_dates=["Date"])
    print(f"Loaded CSV: {path} ({len(df)} rows)")
    return df


# 2) Data Cleaning
def clean_stock_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Clean stock data for a single ticker."""
    df = df.copy().reset_index()
    df["Ticker"] = ticker

    required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' for {ticker}")
    if df.empty:
        raise ValueError(f"No data available for {ticker}")

    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df.sort_values("Date", inplace=True)

    # Handle missing values
    core_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing_pct = df[core_cols].isnull().mean(axis=1)
    df = df[missing_pct < 0.5].ffill().bfill()
    df.drop_duplicates(subset=["Date"], inplace=True)

    # Ensure numeric types
    df[core_cols] = df[core_cols].apply(pd.to_numeric, errors="coerce")

    # Sanity checks
    if (df[["Open", "High", "Low", "Close"]] < 0).any().any():
        raise ValueError(f"Negative prices detected for {ticker}")
    if (df["Volume"] < 0).any():
        raise ValueError(f"Negative volumes detected for {ticker}")

    return df.reset_index(drop=True)


def clean_all_tickers(raw_data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Clean data for all tickers and combine into one DataFrame."""
    cleaned_list = []
    for t in tickers:
        try:
            temp = raw_data[t].reset_index()
            cleaned = clean_stock_data(temp, t)
            cleaned_list.append(cleaned)
            print(f"{t} cleaned successfully ({len(cleaned)} rows).")
        except Exception as e:
            print(f"Error processing {t}: {e}")

    combined = pd.concat(cleaned_list, ignore_index=True)
    return combined.drop(columns=["index"], errors="ignore")


# 3) Trading day alignment
def align_with_trading_days(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Filter data to NYSE trading days only."""
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=start, end_date=end)
    trading_days = schedule.index.normalize()

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    aligned = df[df["Date"].isin(trading_days)]
    print(f"Data aligned with NYSE trading days ({trading_days.min().date()} → {trading_days.max().date()})")
    return aligned


# 4) Outlier detection
def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Add daily returns and detect outliers via Z-score and IQR."""
    df = df.copy()
    df["Daily_Return"] = df.groupby("Ticker")["Close"].pct_change()

    # Z-score method
    df["Z_Score"] = df.groupby("Ticker")["Daily_Return"].transform(
        lambda x: stats.zscore(x, nan_policy="omit")
    )
    df["Outlier_Z"] = df["Z_Score"].abs() > 3

    # IQR method
    def iqr_outliers(series: pd.Series) -> pd.Series:
        q1, q3 = np.nanpercentile(series, [25, 75])
        iqr = q3 - q1
        return (series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))

    df["Outlier_IQR"] = df.groupby("Ticker")["Daily_Return"].transform(iqr_outliers)
    print("Outlier detection completed")
    return df


# 5) Data Quality Report
def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a data quality summary per ticker."""
    core_cols = ["Open", "High", "Low", "Close", "Volume"]
    report = []
    nyse = mcal.get_calendar("NYSE")

    for ticker, group in df.groupby("Ticker"):
        total_rows = len(group)
        missing_pct = (
            group[core_cols].isnull().sum().sum() / (total_rows * len(core_cols))
        ) * 100
        duplicates = group.duplicated(subset=["Date"]).sum()
        min_date, max_date = group["Date"].min(), group["Date"].max()
        outlier_pct = group["Outlier_Z"].mean() * 100 if "Outlier_Z" in group else np.nan
        schedule = nyse.schedule(start_date=min_date, end_date=max_date)
        expected_days = len(schedule)
        coverage_pct = (total_rows / expected_days * 100) if expected_days > 0 else np.nan

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


# 6) Validation
def validate_schema(df: pd.DataFrame, required_cols: list[str] | None = None) -> bool:
    """Validate that required columns exist in the DataFrame."""
    if required_cols is None:
        required_cols = ["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    print("Schema validation passed")
    return True


def validate_preprocessing(df: pd.DataFrame, start_date: str, end_date: str) -> None:
    """Automated checks for preprocessing correctness."""
    assert df[["Open", "High", "Low", "Close", "Volume"]].isnull().sum().sum() == 0, "Missing values remain!"
    for ticker, group in df.groupby("Ticker"):
        expected_returns = group["Close"].pct_change()
        assert np.allclose(group["Daily_Return"].dropna(), expected_returns.dropna(), equal_nan=True), \
            f"Daily returns mismatch for {ticker}"
        assert (group[["Open", "High", "Low", "Close"]] >= 0).all().all(), f"Negative prices detected for {ticker}"
        assert not group["Date"].duplicated().any(), f"Duplicates found in {ticker}"
        assert group["Date"].min() >= pd.to_datetime(start_date), f"Start date invalid for {ticker}"
        assert group["Date"].max() <= pd.to_datetime(end_date), f"End date invalid for {ticker}"
    print("Preprocessing validations passed!")


# 7) Pivot wide format
def pivot_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-format cleaned data into wide format (Close prices)."""
    return df.pivot(index="Date", columns="Ticker", values="Close").ffill().bfill()


# 8) Save outputs
def save_outputs(cleaned_df: pd.DataFrame, pivot_df: pd.DataFrame, config: dict) -> None:
    """Save cleaned datasets (long and wide format)."""
    os.makedirs(os.path.dirname(config["output_long"]), exist_ok=True)
    os.makedirs(os.path.dirname(config["output_wide"]), exist_ok=True)

    cleaned_df.to_csv(config["output_long"], index=False)
    pivot_df.to_csv(config["output_wide"])
    print(f"Datasets saved: {config['output_long']}, {config['output_wide']}")

