"""
data_handler.py

Handles all data-related operations for the Stock Market Trend Analysis project:
- Fetching from Yahoo Finance / CSV
- Cleaning and validation
- Outlier detection and reporting
- Pivoting and saving
- Ensures all data is ready for analytics and visualization modules
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_market_calendars as mcal
from scipy import stats
import logging
from config import CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Display settings
pd.set_option("display.max_columns", None)

logger.info("Packages imported. CONFIG loaded.")


# 1. Data Acquisition
def fetch_stock_data(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Fetch historical stock data from Yahoo Finance.

    Args:
        tickers: List of ticker symbols
        start: Start date string 'YYYY-MM-DD'
        end: End date string 'YYYY-MM-DD'

    Returns:
        DataFrame with fetched data (multi-index if multiple tickers)

    Raises:
        ValueError if no tickers provided or data is empty
    """
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
        logger.info(f"Data fetched successfully for {len(tickers)} tickers ({start} → {end})")
        return data
    except Exception as e:
        logger.error(f"yfinance failed: {e}")
        raise


def load_csv_files(uploaded_files) -> tuple[pd.DataFrame, list[str]]:
    """
    Load one or more CSV files into a combined DataFrame.
    Validates schema, parses dates, assigns unique Ticker names.

    Args:
        uploaded_files: Single file or list of uploaded CSV files

    Returns:
        Tuple (combined DataFrame, list of tickers)

    Raises:
        ValueError if no valid CSVs or missing required columns
    """
    if not uploaded_files:
        raise ValueError("No files uploaded.")
    
    # Ensure uploaded_files is iterable
    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]

    dfs = []
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            df = pd.read_csv(uploaded_file)

            # Validate Date column
            if "Date" not in df.columns:
                logger.warning(f"{uploaded_file.name} skipped — missing 'Date' column.")
                continue

            # Parse and clean Date
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            invalid_dates = df["Date"].isnull().sum()
            if invalid_dates > 0:
                logger.warning(f"{uploaded_file.name}: {invalid_dates} invalid date(s) dropped.")
            df = df.dropna(subset=["Date"]) # Drop rows with invalid dates

            # Assign Ticker if missing
            if "Ticker" not in df.columns:
                ticker_name = f"Stock{i+1}"
                df.insert(0, "Ticker", ticker_name)
                logger.info(f"No 'Ticker' column found in {uploaded_file.name}. Assigned: {ticker_name}")

            dfs.append(df)
            logger.info(f"Loaded CSV: {uploaded_file.name} ({len(df)} rows)")

        except Exception as e:
            logger.error(f"Error loading {uploaded_file.name}: {e}")

    if not dfs:
        raise ValueError("No valid CSV files processed. Please check your files.")

    combined = pd.concat(dfs, ignore_index=True)
    tickers = combined["Ticker"].unique().tolist()

    # Validate schema
    validate_schema(combined)

    # Warn if Close column missing (for SMA calc)
    if "Close" not in combined.columns:
        logger.warning("CSV files missing 'Close' column — SMA calculation may fail.")

    return combined, tickers


# 2. Data Cleaning
def clean_stock_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Clean stock data for a single ticker.

    Args:
        df: Raw stock data
        ticker: Ticker symbol

    Returns:
        Cleaned DataFrame

    Raises:
        ValueError for missing columns, negative values, or empty data
    """
    df = df.copy().reset_index(drop=True)
    df["Ticker"] = ticker

    required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' for {ticker}")
    if df.empty:
        raise ValueError(f"No data available for {ticker}")

    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df.sort_values("Date", inplace=True)

    # Handle missing values (<50% per row) and fill remaining
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
    """
    Clean data for all tickers and combine into one DataFrame.

    Args:
        raw_data: Raw fetched DataFrame
        tickers: List of tickers to process

    Returns:
        Combined cleaned DataFrame
    """
    cleaned_list = []
    for t in tickers:
        try:
            # If raw_data has multi-index (from yf.download), select ticker correctly
            if isinstance(raw_data.columns, pd.MultiIndex):
                temp = raw_data[t].reset_index()
            else:
                temp = raw_data[raw_data["Ticker"] == t].reset_index(drop=True)
            cleaned = clean_stock_data(temp, t)
            cleaned_list.append(cleaned)
            logger.info(f"{t} cleaned successfully ({len(cleaned)} rows).")
        except Exception as e:
            logger.error(f"Error processing {t}: {e}")

    combined = pd.concat(cleaned_list, ignore_index=True)
    return combined.drop(columns=["index"], errors="ignore")


# 3. Trading Day Alignment
def align_with_trading_days(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """
    Filter data to NYSE trading days only.

    Args:
        df: DataFrame with 'Date' column
        start: Start date
        end: End date

    Returns:
        Aligned DataFrame
    """
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=start, end_date=end)
    trading_days = schedule.index.normalize()

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    aligned = df[df["Date"].isin(trading_days)]
    logger.info(f"Data aligned with NYSE trading days ({trading_days.min().date()} → {trading_days.max().date()})")
    return aligned


# 4. Outlier Detection
def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add daily returns and detect outliers using Z-score and IQR methods.

    Args:
        df: Cleaned stock data

    Returns:
        DataFrame with 'Daily_Return', 'Z_Score', 'Outlier_Z', 'Outlier_IQR' columns
    """
    df = df.copy()
    df["Daily_Return"] = df.groupby("Ticker")["Close"].pct_change()

    # Detect outliers using Z-score
    df["Z_Score"] = df.groupby("Ticker")["Daily_Return"].transform(
        lambda x: stats.zscore(x, nan_policy="omit") # Standardize returns
    )
    df["Outlier_Z"] = df["Z_Score"].abs() > 3 # Flag extreme movements

    # Detect outliers using IQR method (vectorized)
    def vectorized_iqr(group):
        q1 = group.quantile(0.25)
        q3 = group.quantile(0.75)
        iqr = q3 - q1
        return (group < (q1 - 1.5 * iqr)) | (group > (q3 + 1.5 * iqr))

    df["Outlier_IQR"] = df.groupby("Ticker")["Daily_Return"].transform(vectorized_iqr)
    logger.info("Outlier detection completed")
    return df


# 5. Data Quality Report
def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a data quality summary per ticker.

    Returns:
        DataFrame with rows, missing %, duplicates, outliers, coverage
    """
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


# 6. Validation
def validate_schema(df: pd.DataFrame, required_cols: list[str] | None = None) -> bool:
    """
    Validate that required columns exist.

    Raises:
        ValueError if columns are missing
    """
    if required_cols is None:
        required_cols = ["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    logger.info("Schema validation passed")
    return True


def validate_preprocessing(df: pd.DataFrame, start_date: str, end_date: str) -> None:
    """
    Automated checks for preprocessing correctness.
    Raises AssertionError on validation failure.
    """
    assert df[["Open", "High", "Low", "Close", "Volume"]].isnull().sum().sum() == 0, "Missing values remain!"
    for ticker, group in df.groupby("Ticker"):
        expected_returns = group["Close"].pct_change()
        assert np.allclose(group["Daily_Return"].dropna(), expected_returns.dropna(), equal_nan=True), \
            f"Daily returns mismatch for {ticker}"
        assert (group[["Open", "High", "Low", "Close"]] >= 0).all().all(), f"Negative prices detected for {ticker}"
        assert not group["Date"].duplicated().any(), f"Duplicates found in {ticker}"
        assert group["Date"].min() >= pd.to_datetime(start_date), f"Start date invalid for {ticker}"
        assert group["Date"].max() <= pd.to_datetime(end_date), f"End date invalid for {ticker}"
    logger.info("Preprocessing validations passed!")


# 7. Pivot Wide Format
def pivot_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot long-format cleaned data into wide format (Close prices).

    Returns:
        Pivoted DataFrame

    Complexity:
        Time: O(n * t) where n = number of rows, t = number of tickers
        Space: O(n * t)
    """
    return df.pivot(index="Date", columns="Ticker", values="Close").ffill().bfill()


# 8. Save Outputs
def save_outputs(cleaned_df: pd.DataFrame, pivot_df: pd.DataFrame, config: dict) -> None:
    """
    Save cleaned datasets (long and wide format).

    Args:
        cleaned_df: Cleaned long-format DataFrame
        pivot_df: Pivoted wide-format DataFrame
        config: Configuration dict with paths
    """
    os.makedirs(os.path.dirname(config["output_long"]), exist_ok=True)
    os.makedirs(os.path.dirname(config["output_wide"]), exist_ok=True)

    cleaned_df.to_csv(config["output_long"], index=False)
    pivot_df.to_csv(config["output_wide"])
    logger.info(f"Datasets saved: {config['output_long']}, {config['output_wide']}")

