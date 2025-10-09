"""
profit.py

Calculates maximum stock trading profits per ticker:
- Single transaction (max profit, optional debug Naive method)
- Multiple transactions (sum of positive differences)
- Applies calculations across multiple tickers
- Returns summary DataFrame or list of dictionaries
"""

import pandas as pd
import numpy as np
import warnings
import logging
from config import CONFIG

# Logger Setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


#  Utility: Debug Mode
def should_keep_debug() -> bool:
    """
    Check whether to keep manual/naive calculations for debugging.

    Returns:
        bool: True if CONFIG['keep_debug'] is enabled.
    """
    return CONFIG.get("keep_debug", False)


# 1. Max Profit (Single Transaction)
def max_profit_single(df: pd.DataFrame, ticker: str, return_indices: bool = False) -> float | tuple:
    """
    Calculate max profit for a single buy-sell transaction.

    Implements:
        - Naive O(n^2) (if keep_debug enabled)
        - Sliding-window O(n) (default efficient)
        - Vectorized O(n) using NumPy

    Args:
        df (pd.DataFrame): Stock dataset with ['Ticker', 'Close', 'Date'].
        ticker (str): Stock ticker symbol.
        return_indices (bool): If True, returns detailed info (profit, buy/sell prices and dates).

    Returns:
        float | tuple: Maximum profit or detailed tuple.

    Example:
        df = pd.DataFrame({"Ticker": ["A"]*5, "Close": [1, 3, 2, 5, 4],
                           "Date": pd.date_range("2020-01-01", periods=5)})
        max_profit_single(df, "A", True)
    """
    required_cols = {"Ticker", "Close", "Date"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must contain {required_cols} columns.")

    subset = df.loc[df["Ticker"] == ticker, ["Close", "Date"]].dropna()
    if len(subset) < 2:
        warnings.warn(f"Not enough data for single transaction profit calculation for {ticker}.")
        return (0.0, 0.0, 0.0, None, None) if return_indices else 0.0

    prices = subset["Close"].values
    dates = subset["Date"].values

    # Debug: Naive O(n^2) Implementation
    if should_keep_debug():
        max_naive = 0.0
        buy_idx_naive = sell_idx_naive = 0
        for i in range(len(prices)):
            for j in range(i + 1, len(prices)):
                profit = prices[j] - prices[i]
                if profit > max_naive:
                    max_naive = profit
                    buy_idx_naive, sell_idx_naive = i, j
        df[f"{ticker}_MaxProfit_Single_Naive"] = max_naive

    # Sliding-window O(n) Implementation
    min_price = prices[0]
    max_profit = 0.0
    buy_price = sell_price = prices[0]
    buy_date = sell_date = dates[0]
    buy_idx = 0

    for i in range(1, len(prices)):
        profit = prices[i] - min_price
        if profit > max_profit:
            max_profit = profit
            buy_price = min_price
            sell_price = prices[i]
            buy_date = dates[buy_idx]
            sell_date = dates[i]
        if prices[i] < min_price:
            min_price = prices[i]
            buy_idx = i

    # Vectorized O(n) (NumPy) [optional reference]
    min_prices = np.minimum.accumulate(prices[:-1])
    vectorized_profit = np.max(prices[1:] - min_prices)

    if return_indices:
        return max_profit, buy_price, sell_price, buy_date, sell_date
    return float(max_profit)


# 2. Max Profit (Multiple Transactions)
def max_profit_multiple(df: pd.DataFrame, ticker: str) -> float:
    """
    Calculate max profit with unlimited buy-sell transactions.

    Algorithm:
        - Sum all positive consecutive differences (vectorized O(n))

    Args:
        df (pd.DataFrame): Stock dataset with ['Ticker', 'Close'].
        ticker (str): Stock ticker symbol.

    Returns:
        float: Maximum profit achievable with multiple transactions.

    Example:
        df = pd.DataFrame({"Ticker": ["A"]*5, "Close": [1,3,2,5,4]})
        max_profit_multiple(df, "A")  # returns 5
    """
    if not {"Ticker", "Close"}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'Ticker' and 'Close' columns.")

    prices = df.loc[df["Ticker"] == ticker, "Close"].dropna().values
    if len(prices) < 2:
        warnings.warn(f"Not enough data for multiple transaction profit calculation for {ticker}.")
        return 0.0

    profit = np.sum(np.diff(prices)[np.diff(prices) > 0])
    return float(profit)


# 3. Apply Profit Calculations to All Tickers
def calculate_profits(df: pd.DataFrame, tickers: list[str] = None, as_dict: bool = False):
    """
    Apply single and multiple transaction profit calculations for multiple tickers.

    Adds (if debug enabled):
        - MaxProfit_Single_Naive (O(n^2) baseline)
    Calculates:
        - MaxProfit_Single
        - MaxProfit_Multiple
        - Return_Pct (from best single trade)
        - Best buy/sell prices and dates

    Args:
        df (pd.DataFrame): Stock dataset with ['Ticker', 'Close', 'Date'].
        tickers (list[str], optional): Tickers to process. Defaults to CONFIG['tickers'] or unique tickers.
        as_dict (bool, optional): If True, returns a list of dicts instead of DataFrame.

    Returns:
        pd.DataFrame | list[dict]: Profit summary per ticker.
    """
    required_cols = {"Ticker", "Close", "Date"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must contain {required_cols} columns.")

    tickers = tickers or CONFIG.get("tickers", df["Ticker"].unique())
    results = []
    grouped = df.groupby("Ticker")

    for ticker in tickers:
        try:
            group = grouped.get_group(ticker) if ticker in grouped.groups else pd.DataFrame()
            single, buy_price, sell_price, buy_date, sell_date = max_profit_single(
                group, ticker, return_indices=True
            )
            multiple = max_profit_multiple(group, ticker)

            # Percentage return from single trade
            return_pct = (single / buy_price * 100) if buy_price > 0 else 0.0

            result = {
                "Ticker": ticker,
                "Best_Buy_Date": buy_date,
                "Best_Buy_Price": buy_price,
                "Best_Sell_Date": sell_date,
                "Best_Sell_Price": sell_price,
                "MaxProfit_Single": single,
                "Return_Pct": return_pct,
                "MaxProfit_Multiple": multiple,
            }

            # Include naive debug result if available
            if should_keep_debug():
                naive_col = f"{ticker}_MaxProfit_Single_Naive"
                if naive_col in group.columns:
                    result["MaxProfit_Single_Naive"] = float(group[naive_col].iloc[0])

            results.append(result)

            # Format dates nicely
            buy_date_str = pd.to_datetime(buy_date).strftime("%Y-%m-%d") if pd.notnull(buy_date) else "-"
            sell_date_str = pd.to_datetime(sell_date).strftime("%Y-%m-%d") if pd.notnull(sell_date) else "-"

            # Logging
            if CONFIG.get("enable_logging", True):
                logger.info(
                    f"{ticker} → Buy: {buy_date_str} @ ${buy_price:.2f}, "
                    f"Sell: {sell_date_str} @ ${sell_price:.2f}, "
                    f"Profit: ${single:.2f}, Return: {return_pct:.2f}%, "
                    f"MultiTx: ${multiple:.2f}"
                )

        except Exception as e:
            if CONFIG.get("enable_logging", True):
                logger.error(f"Error calculating profits for {ticker}: {e}")

    if CONFIG.get("enable_logging", True):
        logger.info("Profit optimization completed for all tickers.\n")

    return results if as_dict else pd.DataFrame(results)
