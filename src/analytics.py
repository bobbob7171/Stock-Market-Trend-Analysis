"""
Analytics module for Stock Market Trend Analysis.

Contains functions for:
- Daily returns calculation
- Simple Moving Average (SMA)
- Streak detection
- Annual risk-return computation
"""

import pandas as pd
import numpy as np
import logging
from config import CONFIG  # Project configuration

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def should_keep_debug() -> bool:
    """
    Check if debug/manual/vectorized columns should be kept.

    Returns:
        bool: True if debug is enabled in CONFIG, else False
    """
    return CONFIG.get("keep_debug", False)

#1. Daily Returns Calculation
def calculate_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily returns per ticker.

    Adds:
        - 'Daily_Return_manual' (if debug enabled)
        - 'Daily_Return' (default efficient version)

    Args:
        df (pd.DataFrame): Must contain ['Ticker', 'Close']

    Returns:
        pd.DataFrame: Original DataFrame with added daily return columns
    """
    df = df.copy()

    if should_keep_debug():
        # Manual loop version for debugging
        for ticker, group in df.groupby("Ticker"):
            closes = group["Close"].tolist()
            returns = [np.nan]
            for i in range(1, len(closes)):
                prev, curr = closes[i - 1], closes[i]
                returns.append((curr - prev) / prev if prev != 0 else np.nan)
            df.loc[group.index, "Daily_Return_manual"] = returns

    # Efficient vectorized pandas version
    df["Daily_Return"] = df.groupby("Ticker")["Close"].pct_change()

    if CONFIG.get("enable_logging", True):
        logger.info("Daily returns calculated.")
    return df

#2. Simple Moving Average (SMA) Calculation
def calculate_sma(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    """
    Compute Simple Moving Averages (SMA) for multiple window sizes per ticker.

    Adds:
        - SMA_{window}_manual   : Naive O(n·k) implementation (for clarity/debug)
        - SMA_{window}_sliding  : Sliding window O(n) implementation (more efficient)
        - SMA_{window}          : Efficient pandas rolling (default)

    Args:
        df (pd.DataFrame): Must contain ['Ticker', 'Close']
        windows (list[int], optional): SMA window sizes. Defaults to CONFIG['sma_windows']

    Returns:
        pd.DataFrame: DataFrame with SMA columns added
    """
    df = df.copy()
    if windows is None:
        windows = CONFIG.get("sma_windows", [20, 50])

    for window in windows:
        if window < 1:
            raise ValueError("SMA window must be at least 1.")

        if should_keep_debug():
            manual_col = f"SMA_{window}_manual"
            sliding_col = f"SMA_{window}_sliding"

            for ticker, group in df.groupby("Ticker"):
                closes = group["Close"].tolist()

                # Naive O(n·k) manual computation
                sma_naive = [
                    np.nan if i < window - 1 else np.mean(closes[i - window + 1: i + 1])
                    for i in range(len(closes))
                ]

                # Sliding window O(n) computation
                window_sum = 0
                sma_sliding = []
                for i in range(len(closes)):
                    window_sum += closes[i]
                    if i >= window:
                        window_sum -= closes[i - window]
                    sma_sliding.append(window_sum / window if i >= window - 1 else np.nan)

                df.loc[group.index, manual_col] = sma_naive
                df.loc[group.index, sliding_col] = sma_sliding

        # Efficient pandas rolling version
        df[f"SMA_{window}"] = df.groupby("Ticker")["Close"].transform(
            lambda x: x.rolling(window=window, min_periods=window).mean()
        )

    if CONFIG.get("enable_logging", True):
        logger.info(f"SMA calculated for windows: {windows}")
    return df

#3. Streak Detection
def detect_streaks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect consecutive positive/negative daily return streaks.

    Adds:
        - 'Streak_manual', 'Streak_numpy' (if debug enabled)
        - 'Streak_pandas' (internal)
        - 'Streak' (default column)

    Args:
        df (pd.DataFrame): Must contain ['Ticker', 'Daily_Return']

    Returns:
        pd.DataFrame: DataFrame with streak columns
    """
    df = df.copy()

    if should_keep_debug():
        # Manual loop streak detection
        def manual_streak(series):
            streaks = []
            current = 0
            for ret in series:
                if pd.isna(ret):
                    streaks.append(np.nan)
                    current = 0
                elif ret > 0:
                    current = current + 1 if current >= 0 else 1
                    streaks.append(current)
                elif ret < 0:
                    current = current - 1 if current <= 0 else -1
                    streaks.append(current)
                else:
                    streaks.append(0)
                    current = 0
            return streaks

        df["Streak_manual"] = df.groupby("Ticker")["Daily_Return"].transform(manual_streak)

        # Numpy vectorized version
        def numpy_streak(series):
            arr = np.sign(series.fillna(0).to_numpy())
            streaks = np.zeros_like(arr, dtype=float)
            current = 0
            mask_nan = series.isna().to_numpy()
            for i in range(len(arr)):
                if mask_nan[i]:
                    streaks[i] = np.nan
                    current = 0
                elif arr[i] == 0:
                    streaks[i] = 0
                    current = 0
                elif arr[i] > 0:
                    current = current + 1 if current >= 0 else 1
                    streaks[i] = current
                else:
                    current = current - 1 if current <= 0 else -1
                    streaks[i] = current
            return streaks

        df["Streak_numpy"] = df.groupby("Ticker")["Daily_Return"].transform(numpy_streak)

    # Pandas-efficient default streak computation
    df["Direction"] = np.sign(df["Daily_Return"])
    streak_values = []
    for ticker, group in df.groupby("Ticker", sort=False):
        dir_vals = group["Direction"].to_numpy()
        streak = np.full_like(dir_vals, np.nan, dtype=float)
        current = 0
        for i, d in enumerate(dir_vals):
            if np.isnan(d):
                current = 0
            elif d == 0:
                streak[i] = 0
                current = 0
            elif d > 0:
                current = current + 1 if current >= 0 else 1
                streak[i] = current
            else:
                current = current - 1 if current <= 0 else -1
                streak[i] = current
        streak_values.extend(streak)

    df["Streak_pandas"] = streak_values
    df["Streak"] = df["Streak_pandas"]

    # Clean up temporary columns if not in debug
    if not should_keep_debug():
        df.drop(columns=["Streak_pandas", "Direction"], inplace=True)

    if CONFIG.get("enable_logging", True):
        logger.info("Streaks detected. Default 'Streak' column created.")

    return df

#4. Annual Risk-Return Calculation
def calculate_annual_risk_return(
    df: pd.DataFrame, risk_free_rate: float = 0.03, trading_days: int = 252
) -> pd.DataFrame:
    """
    Calculate annualized return, volatility, and Sharpe ratio per ticker.

    Args:
        df (pd.DataFrame): Must contain ['Ticker', 'Daily_Return']
        risk_free_rate (float): Annual risk-free rate (default 0.03)
        trading_days (int): Trading days per year (default 252)

    Returns:
        pd.DataFrame: DataFrame with ['Ticker', 'Annual_Return', 'Annual_Volatility', 'Sharpe_Ratio']
    """
    summary = []

    for ticker, group in df.groupby("Ticker"):
        daily_ret = group["Daily_Return"].dropna()
        if daily_ret.empty:
            continue
        annual_return = daily_ret.mean() * trading_days
        annual_volatility = daily_ret.std() * np.sqrt(trading_days)
        sharpe_ratio = ((annual_return - risk_free_rate) / annual_volatility
                        if annual_volatility != 0 else 0)
        summary.append({
            "Ticker": ticker,
            "Annual_Return": annual_return,
            "Annual_Volatility": annual_volatility,
            "Sharpe_Ratio": sharpe_ratio
        })

    risk_return_df = pd.DataFrame(summary)
    if CONFIG.get("enable_logging", True):
        logger.info("Annual risk-return calculated.")
    return risk_return_df
