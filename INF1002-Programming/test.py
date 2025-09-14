import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import json
import os

# ------------------------------
# Data Loading
# ------------------------------
def load_data(file_path="data/AAPL_stock.csv"):
    """
    Load and preprocess stock data.
    Returns DataFrame or raises an error if file is invalid.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_csv(file_path, skiprows=2)
        df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df[["Open", "High", "Low", "Close", "Volume"]] = df[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce")

        df.dropna(inplace=True)
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)

        return df

    except Exception as e:
        print(f"Error loading data: {e}")
        return None


# ------------------------------
# Core Functions
# ------------------------------
def compute_daily_returns(df, column='Close'):
    """Compute simple daily returns."""
    df['Return'] = df[column].pct_change()
    return df

def compute_sma(df, window=5, column='Close'):
    """Compute Simple Moving Average (SMA)."""
    sma_column = f'SMA_{window}'
    df[sma_column] = df[column].rolling(window=window).mean()
    return df

def compute_runs(df, column='Close'):
    """
    Compute up/down runs, total days, longest streaks.
    Returns stats dict + df with run markers.
    """
    df['Change'] = df[column].diff()
    df['Direction'] = np.sign(df['Change'])
    df['Group'] = (df['Direction'] != df['Direction'].shift()).cumsum()

    groups = df.groupby('Group')
    up_runs, down_runs = [], []

    for _, g in groups:
        if g['Direction'].iloc[0] == 1:
            up_runs.append(len(g))
        elif g['Direction'].iloc[0] == -1:
            down_runs.append(len(g))

    stats = {
        'num_up_runs': len(up_runs),
        'num_down_runs': len(down_runs),
        'total_up_days': sum(up_runs),
        'total_down_days': sum(down_runs),
        'longest_up_streak': max(up_runs) if up_runs else 0,
        'longest_down_streak': max(down_runs) if down_runs else 0
    }
    return stats, df

def compute_max_profit(prices):
    """Compute max profit with multiple transactions allowed."""
    return sum(max(0, prices[i] - prices[i-1]) for i in range(1, len(prices)))


# ------------------------------
# Visualization
# ------------------------------
def plot_price_and_sma(df, column='Close', sma_column='SMA_5'):
    """Plot price vs SMA."""
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[column], label='Closing Price')
    plt.plot(df.index, df[sma_column], label=sma_column)
    plt.title('Closing Price vs. SMA')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig('outputs/price_vs_sma.png')
    plt.close()

def plot_highlighted_runs(df, column='Close'):
    """Plot runs with colors."""
    points = np.array([df.index, df[column]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    colors = [
        'green' if d == 1 else 'red' if d == -1 else 'black'
        for d in df['Direction'].iloc[1:]
    ]

    lc = LineCollection(segments, colors=colors, linewidths=2)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_title('Closing Price with Up/Down Runs Highlighted')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    plt.grid(True)
    plt.savefig('outputs/highlighted_runs.png')
    plt.close()


# ------------------------------
# Results Export
# ------------------------------
def save_results(runs_stats, max_profit, file="outputs/results.json"):
    os.makedirs("outputs", exist_ok=True)
    results = {
        "runs_statistics": runs_stats,
        "maximum_profit": max_profit
    }
    with open(file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results exported to {file}")


# ------------------------------
# Test Cases
# ------------------------------
def run_tests():
    """Run 5 test cases to validate correctness."""
    print("\nRunning Validation Tests...")
    tests = {
        "constant_prices": [100, 100, 100, 100],
        "strictly_increasing": [1, 2, 3, 4, 5],
        "strictly_decreasing": [5, 4, 3, 2, 1],
        "zigzag": [1, 3, 2, 4, 3, 5],
        "realistic_small": [100, 102, 101, 105, 104]
    }
    for name, prices in tests.items():
        profit = compute_max_profit(prices)
        print(f"Test {name}: Max Profit = {profit}")


# ------------------------------
# CLI Menu
# ------------------------------
def main():
    print("=== Stock Market Trend Analysis ===")
    file_path = input("Enter CSV file path (default: data/AAPL_stock.csv): ") or "data/AAPL_stock.csv"
    df = load_data(file_path)
    if df is None:
        return

    window = int(input("Enter SMA window size (default 5): ") or 5)

    # Compute metrics
    df = compute_sma(df, window=window)
    df = compute_daily_returns(df)
    runs_stats, df = compute_runs(df)
    max_profit = compute_max_profit(df['Close'].tolist())

    # Display results
    print("\n--- Analysis Results ---")
    print("Runs Statistics:", runs_stats)
    print("Maximum Profit:", max_profit)
    print("Sample Daily Returns:\n", df['Return'].head())

    # Save + Plot
    save_results(runs_stats, max_profit)
    plot_price_and_sma(df, sma_column=f'SMA_{window}')
    plot_highlighted_runs(df)

    # Run validation tests
    run_tests()


if __name__ == "__main__":
    main()
