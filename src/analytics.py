import pandas as pd
import numpy as np
from config import CONFIG  # Import project config


def should_keep_debug() -> bool:
    """Helper toggle to decide if debug/manual/vectorized cols are kept."""
    return CONFIG.get("keep_debug", False)


def calculate_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily returns per ticker.

    Adds:
        - 'Daily_Return_manual' (if keep_debug enabled)
        - 'Daily_Return' (default efficient version)
    """
    df = df.copy()

    if should_keep_debug():
        for ticker, group in df.groupby("Ticker"):
            closes = group["Close"].tolist()
            returns = [np.nan]
            for i in range(1, len(closes)):
                prev, curr = closes[i - 1], closes[i]
                ret = (curr - prev) / prev if prev != 0 else np.nan
                returns.append(ret)
            df.loc[group.index, "Daily_Return_manual"] = returns

    # Efficient pandas version
    df["Daily_Return"] = df.groupby("Ticker")["Close"].pct_change()

    if CONFIG.get("enable_logging", True):
        print("Daily returns calculated.")
    return df


def calculate_sma(df: pd.DataFrame, windows=None) -> pd.DataFrame:
    """
    Compute SMA (Simple Moving Average) per ticker.

    Adds:
        - SMA_{window}_manual (if keep_debug enabled)
        - SMA_{window} (efficient pandas version)
    """
    df = df.copy()
    if windows is None:
        windows = CONFIG.get("sma_windows", [20, 50])

    for window in windows:
        if should_keep_debug():
            manual_col = f"SMA_{window}_manual"
            for ticker, group in df.groupby("Ticker"):
                closes = group["Close"].tolist()
                sma_vals = []
                for i in range(len(closes)):
                    if i < window - 1:
                        sma_vals.append(np.nan)
                    else:
                        sma_vals.append(np.mean(closes[i - window + 1 : i + 1]))
                df.loc[group.index, manual_col] = sma_vals

        # Efficient pandas version
        pandas_col = f"SMA_{window}"
        df[pandas_col] = df.groupby("Ticker")["Close"].transform(
            lambda x: x.rolling(window=window).mean()
        )

    if CONFIG.get("enable_logging", True):
        print(f"SMA calculated for windows: {windows}")

    return df


def detect_streaks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect streaks of consecutive positive/negative returns.

    Adds:
        - 'Streak_manual' and 'Streak_numpy' (if keep_debug enabled)
        - 'Streak_pandas' (efficient)
        - 'Streak' (default, used in pipeline)
    """
    df = df.copy()

    if should_keep_debug():
        # Manual version
        for ticker, group in df.groupby("Ticker"):
            streaks = []
            current_streak = 0
            for ret in group["Daily_Return"].tolist():
                if pd.isna(ret):
                    streaks.append(np.nan)
                    continue
                if ret > 0:
                    current_streak = current_streak + 1 if current_streak >= 0 else 1
                elif ret < 0:
                    current_streak = current_streak - 1 if current_streak <= 0 else -1
                else:
                    current_streak = 0
                streaks.append(current_streak)
            df.loc[group.index, "Streak_manual"] = streaks

        # Vectorized numpy transform version
        def vectorized_streak(series):
            arr = np.sign(series.fillna(0).to_numpy())
            streaks = np.zeros_like(arr, dtype=float)
            current = 0
            for i in range(len(arr)):
                if arr[i] == 0:
                    streaks[i] = 0
                    current = 0
                elif arr[i] > 0:
                    current = current + 1 if current >= 0 else 1
                    streaks[i] = current
                else:
                    current = current - 1 if current <= 0 else -1
                    streaks[i] = current
            streaks[np.isnan(series.to_numpy())] = np.nan
            return streaks

        df["Streak_numpy"] = df.groupby("Ticker")["Daily_Return"].transform(vectorized_streak)

    # Efficient pandas version
    df["Direction"] = np.sign(df["Daily_Return"])
    df["Group"] = df.groupby("Ticker")["Direction"].diff().ne(0).cumsum()
    df["Streak_pandas"] = df.groupby(["Ticker", "Group"]).cumcount() + 1
    df.loc[df["Direction"] < 0, "Streak_pandas"] *= -1
    df.loc[df["Direction"] == 0, "Streak_pandas"] = 0
    df.drop(columns=["Group", "Direction"], inplace=True)

    df["Streak"] = df["Streak_pandas"]

    if not should_keep_debug():
        df.drop(columns=["Streak_pandas"], inplace=True)

    if CONFIG.get("enable_logging", True):
        print("Streaks detected. Default 'Streak' column created.")

    return df
