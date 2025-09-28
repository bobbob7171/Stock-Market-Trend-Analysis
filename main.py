# main.py

import os
import logging
import argparse
import pandas as pd
from tabulate import tabulate

# Import CONFIG directly from config.py (project root)
from config import CONFIG

# src modules
from src.data_handler import (
    fetch_stock_data, clean_all_tickers, align_with_trading_days,
    detect_outliers, data_quality_report, validate_preprocessing,
    pivot_wide, save_outputs, load_csv, validate_schema
)
from src.analytic import calculate_daily_returns, calculate_sma, detect_streaks
from src.profit import calculate_profits
from src.graph import (
    plot_price_sma, plot_streaks, plot_profit_comparison
)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# Helper Functions
def print_table(df: pd.DataFrame, title: str = "Table") -> None:
    """Nicely format and print a DataFrame using tabulate."""
    if df.empty:
        logging.warning(f"{title}: DataFrame is empty, nothing to display.")
        return

    df_formatted = df.copy()
    for col in df_formatted.select_dtypes(include=["float", "int"]).columns:
        df_formatted[col] = df_formatted[col].map("{:.2f}".format)

    logging.info(f"\n{title}\n{tabulate(df_formatted, headers='keys', tablefmt='pretty')}")


def save_csv(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to CSV, ensuring the directory exists."""
    if df.empty:
        logging.warning(f"Skipped saving {path} (DataFrame empty).")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logging.info(f"Saved: {path}")


def get_user_input():
    """Collect tickers, date range, and SMA windows from user input (defaults from CONFIG)."""
    tickers_input = input(f"Enter tickers (comma-separated) or press Enter for default {CONFIG['tickers']}: ")
    tickers = [t.strip().upper() for t in tickers_input.split(",")] if tickers_input else CONFIG["tickers"]

    start_date = input(f"Enter start date (YYYY-MM-DD) or press Enter for default {CONFIG['start_date']}: ").strip() \
        or CONFIG["start_date"]
    end_date = input(f"Enter end date (YYYY-MM-DD) or press Enter for default {CONFIG['end_date']}: ").strip() \
        or CONFIG["end_date"]

    sma_input = input(f"Enter SMA windows (comma-separated) or press Enter for default {CONFIG['sma_windows']}: ")
    if sma_input:
        sma_windows = [int(x.strip()) for x in sma_input.split(",") if x.strip().isdigit() and int(x) > 0]
        if not sma_windows:
            sma_windows = CONFIG["sma_windows"]
    else:
        sma_windows = CONFIG["sma_windows"]

    return tickers, start_date, end_date, sma_windows


def summarize_streaks(df: pd.DataFrame) -> pd.DataFrame:
    """Compute streak summary statistics per ticker."""
    if "Streak" not in df.columns or "Daily_Return" not in df.columns:
        logging.warning("Missing required columns for streak summary.")
        return pd.DataFrame()

    summaries = []
    for ticker, group in df.groupby("Ticker"):
        streaks = group["Streak"].dropna()
        if streaks.empty:
            continue

        prev = streaks.shift(1).fillna(0)
        summaries.append({
            "Ticker": ticker,
            "Highest_Up_Streak": streaks[streaks > 0].max(skipna=True) or 0,
            "Highest_Down_Streak": streaks[streaks < 0].min(skipna=True) or 0,
            "Num_Up_Streaks": ((streaks > 0) & (prev <= 0)).sum(),
            "Num_Down_Streaks": ((streaks < 0) & (prev >= 0)).sum(),
            "Total_Up_Days": (group["Daily_Return"] > 0).sum(),
            "Total_Down_Days": (group["Daily_Return"] < 0).sum(),
        })

    return pd.DataFrame(summaries)


# Core Pipeline
def run_data_pipeline(tickers, start, end, backup_path):
    """Fetch, clean, align, validate, and pivot stock data."""
    raw_data = fetch_stock_data(tickers, start, end, backup_path)
    cleaned = clean_all_tickers(raw_data, tickers)
    cleaned = align_with_trading_days(cleaned, start, end)
    cleaned = detect_outliers(cleaned)

    validate_schema(cleaned)
    validate_preprocessing(cleaned, start, end)

    print_table(data_quality_report(cleaned), title="Data Quality Report")
    pivot = pivot_wide(cleaned)
    logging.info(f"Wide-format data shape: {pivot.shape}")
    return cleaned, pivot


def run_analytics(cleaned_data, sma_windows):
    """Run core analytics: Daily Returns, SMA, Streaks."""
    if cleaned_data.empty:
        logging.warning("No cleaned data available. Skipping analytics.")
        return pd.DataFrame()

    analytics_data = calculate_daily_returns(cleaned_data)
    analytics_data = calculate_sma(analytics_data, windows=sma_windows)
    analytics_data = detect_streaks(analytics_data)

    logging.info(f"Core analytics completed (Daily Returns, SMA {sma_windows}, Streaks).")
    return analytics_data


def run_profit(analytics_data, tickers):
    """Run profit calculation using sliding-window approach and print summary."""
    if analytics_data.empty:
        logging.warning("No analytics data available. Skipping profit calculation.")
        return pd.DataFrame()

    logging.info("Calculating profits using sliding-window method (single & multiple transactions).")
    profit_summary = calculate_profits(analytics_data, tickers)
    print_table(profit_summary, title="Profit Summary")
    return profit_summary


# Main Entry
def main(no_input: bool = False, plot: bool = False):
    logging.info("🔹 Welcome to Stock Analytics 🔹")

    # Data input
    if no_input:
        tickers, start_date, end_date, sma_windows = (
            CONFIG["tickers"], CONFIG["start_date"], CONFIG["end_date"], CONFIG["sma_windows"]
        )
        cleaned_data, pivot_data = run_data_pipeline(tickers, start_date, end_date, CONFIG["backup_path"])
    else:
        if input("Load data from CSV instead of yfinance? (y/n, default n): ").lower() == "y":
            csv_path = input(f"Enter CSV path (default: {CONFIG['backup_path']}): ").strip() or CONFIG["backup_path"]
            cleaned_data = load_csv(csv_path)
            validate_schema(cleaned_data)
            pivot_data = pivot_wide(cleaned_data)
            sma_windows = CONFIG["sma_windows"]
        else:
            tickers, start_date, end_date, sma_windows = get_user_input()
            cleaned_data, pivot_data = run_data_pipeline(tickers, start_date, end_date, CONFIG["backup_path"])

    # Save processed datasets
    save_outputs(cleaned_data, pivot_data, CONFIG)

    # Analytics
    analytics_data = run_analytics(cleaned_data, sma_windows)
    if not analytics_data.empty:
        save_csv(analytics_data, CONFIG["analytics_path"])

        # Export wide daily returns
        returns_wide = analytics_data.pivot(index="Date", columns="Ticker", values="Daily_Return")
        save_csv(returns_wide.reset_index(), CONFIG["returns_wide_path"])

        # Export streaks
        save_csv(analytics_data[["Date", "Ticker", "Streak"]], CONFIG["streaks_path"])

        # Streak summary
        streak_summary = summarize_streaks(analytics_data)
        print_table(streak_summary, title="Streak Summary")
        save_csv(streak_summary, CONFIG["streak_summary_path"])

        # Plot
        if plot or (not no_input and input("Plot closing prices and SMAs? (y/n, default y): ").lower() != "n"):
            for ticker in analytics_data["Ticker"].unique():
                plot_price_sma(analytics_data, ticker, sma_windows=sma_windows)
                plot_streaks(analytics_data, ticker)

    # Profit calculations
    profit_summary = run_profit(analytics_data, list(cleaned_data["Ticker"].unique()))
    if not profit_summary.empty:
        save_csv(profit_summary, CONFIG["profit_summary_path"])
        if plot or (not no_input and input("Plot profit summary? (y/n, default y): ").lower() != "n"):
            plot_profit_comparison(profit_summary)

    logging.info("All selected steps completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Analytics Pipeline")
    parser.add_argument("--no-input", action="store_true", help="Run with defaults (non-interactive mode).")
    parser.add_argument("--plot", action="store_true", help="Plot results even in non-interactive mode.")
    args = parser.parse_args()

    main(no_input=args.no_input, plot=args.plot)