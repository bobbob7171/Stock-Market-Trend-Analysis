# profit.py

import pandas as pd
import numpy as np
import warnings
from config import CONFIG


# 1. Max Profit (Single Transaction, sliding-window style)
def max_profit_single(df: pd.DataFrame, ticker: str) -> float:
    """
    Calculate max profit for a single buy-sell transaction using a sliding-window approach.

    Algorithm:
    - Track the minimum price seen so far (start of window)
    - For each price, calculate profit if sold at current price
    - Update max profit if current profit is higher
    - Sliding min window moves forward implicitly

    Args:
        df (pd.DataFrame): Stock dataset with ['Ticker', 'Close'].
        ticker (str): Stock ticker symbol.

    Returns:
        float: Maximum profit achievable with one transaction.
    """
    if not {"Ticker", "Close"}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'Ticker' and 'Close' columns")

    prices = df.loc[df["Ticker"] == ticker, "Close"].values
    if len(prices) < 2:
        warnings.warn(f"Not enough data to calculate single transaction profit for {ticker}.")
        return 0.0

    min_price, max_profit = prices[0], 0.0
    for price in prices[1:]:
        max_profit = max(max_profit, price - min_price)
        min_price = min(min_price, price)

    return float(max_profit)


# 2. Max Profit (Multiple Transactions, sliding-window style)
def max_profit_multiple(df: pd.DataFrame, ticker: str) -> float:
    """
    Calculate maximum profit with unlimited transactions using a sliding-window approach.

    Algorithm:
    - For each consecutive pair, add the positive difference (profit)
    - Skip negative differences (simulate no transaction)
    - Stepwise addition mimics sliding window of gains

    Args:
        df (pd.DataFrame): Stock dataset with ['Ticker', 'Close'].
        ticker (str): Stock ticker symbol.

    Returns:
        float: Maximum profit achievable with multiple transactions.
    """
    if not {"Ticker", "Close"}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'Ticker' and 'Close' columns")

    prices = df.loc[df["Ticker"] == ticker, "Close"].values
    if len(prices) < 2:
        warnings.warn(f"Not enough data to calculate multiple transaction profit for {ticker}.")
        return 0.0

    profit = 0.0
    for i in range(1, len(prices)):
        gain = prices[i] - prices[i - 1]
        if gain > 0:
            profit += gain  # add gain only if positive

    return float(profit)


# 3. Apply to All Tickers
def calculate_profits(df: pd.DataFrame, tickers: list[str] = None, as_dict: bool = False):
    """
    Run both single and multiple transaction profit strategies across tickers.

    Args:
        df (pd.DataFrame): Processed stock dataset.
        tickers (list[str], optional): List of stock tickers. Defaults to CONFIG["tickers"].
        as_dict (bool, optional): If True, return results as dict instead of DataFrame.

    Returns:
        pd.DataFrame | list[dict]: Profit summary with single vs multiple strategies.
    """
    if "Ticker" not in df.columns:
        raise ValueError("DataFrame must contain 'Ticker' column")

    tickers = tickers or CONFIG["tickers"]
    results = []

    for ticker in tickers:
        try:
            single = max_profit_single(df, ticker)
            multiple = max_profit_multiple(df, ticker)

            if CONFIG.get("enable_logging", True):
                print(f"{ticker} → Single Tx Profit: {single:.2f}, Multi Tx Profit: {multiple:.2f}")

            results.append({
                "Ticker": ticker,
                "MaxProfit_Single": single,
                "MaxProfit_Multiple": multiple
            })
        except Exception as e:
            if CONFIG.get("enable_logging", True):
                print(f"Error calculating profits for {ticker}: {e}")

    if CONFIG.get("enable_logging", True):
        print("Profit optimization completed for all tickers.\n")

    return results if as_dict else pd.DataFrame(results)

def get_buy_sell_trades(df: pd.DataFrame, ticker: str) -> list[tuple[int, int]]:
    """
    Generate buy/sell indices for maximum profit using multiple transactions.
    Returns a list of (buy_index, sell_index) tuples.
    """
    prices = df.loc[df["Ticker"] == ticker, "Close"].values
    trades = []
    i = 0
    while i < len(prices) - 1:
        # Find local minima (buy)
        while i < len(prices) - 1 and prices[i + 1] <= prices[i]:
            i += 1
        buy = i
        i += 1
        # Find local maxima (sell)
        while i < len(prices) and prices[i] >= prices[i - 1]:
            i += 1
        sell = i - 1
        if buy < sell:
            trades.append((buy, sell))
    return trades
