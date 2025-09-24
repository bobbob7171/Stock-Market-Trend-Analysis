import pandas as pd
def daily_returns(prices:list[float])-> list[float | None]:
    """
    Calculate daily returns from a list of prices.

    Formula:
    Daily Return = (Price_today - Price_yesterday) / Price_yesterday

    Parameters:
    prices (list of float): List of daily prices.

    Returns:
    list of float: List of daily returns.
    """

    n = len(prices)
    if n == 0: # check is the length if price is zero it will return none
        return[]
    returns = [None] * n

    for i in range(1, n):
        if prices[i - 1] != 0: # checks the previous day prices and ensures it is'nt zero
            returns[i] = (prices[i] - prices[i - 1]) / prices[i - 1] # todays price - yesterday price / yesterdays price
        else:
            returns[i] = None # since yesterday price is 0 it cant compute the return
    return returns

if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv('data/cleaned_stock_data_wide.csv')
    date_column = df.columns[0]
    stocks = list(df.columns[1:])

    # --- ADD: prepare and export daily-returns datasets for plotting ---

# Ensure dates are datetime and sorted (needed for pct_change/rolling)
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    df = df.sort_values(date_column).reset_index(drop=True)

    tickers = stocks  # all price columns

# (1) WIDE daily returns matrix -> for correlation heatmap
    returns_wide = df[tickers].pct_change()
    returns_wide.insert(0, "Date", df[date_column].values)
    returns_wide.to_csv("daily_returns_wide.csv", index=False)

# (2) LONG tidy table with Close, DailyReturn, and 30-day rolling volatility
    long_close = (
        df[[date_column] + tickers]
        .melt(id_vars=[date_column], var_name="Ticker", value_name="Close")
        .sort_values(["Ticker", date_column])
        )
    long_close["DailyReturn"] = long_close.groupby("Ticker")["Close"].pct_change()
    long_close["RollingVol_30"] = (
        long_close.groupby("Ticker")["DailyReturn"]
        .rolling(30).std().reset_index(level=0, drop=True)
        )
    long_close.rename(columns={date_column: "Date"}).to_csv("daily_returns_long.csv", index=False)

    # Ask user to select stock
    print("Available stocks:",", ".join(stocks))
    stock_choice = input("Enter stock symbol: ").strip().upper()

    if stock_choice not in stocks:
        raise ValueError(f"Stock '{stock_choice}' not found in dataset.")

    try:
         # Extract close prices
        prices = df[stock_choice].tolist()

        d_returns = daily_returns(prices)

        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

        dates = df[date_column].dt.strftime("%Y-%m-%d").tolist()

        max_days = len(prices) - 1

        days_count = int(input(f"Enter number of days to display (max {max_days}): "))

        if days_count < 1 or days_count > max_days:
            raise ValueError(f"Days count must be between 1 and {max_days}.")
        print("\n===== Stock Daily Returns Analysis =====")
        print(f"First {days_count} Daily Returns:")
        print("---------------------------------------------")
        for i in range(1, days_count + 1):
            print(f"Date: {dates[i]} Price: ${prices[i]:,.2f} Daily Return: {d_returns[i]:.4%}")
    except FileNotFoundError:
        print("Could not find Clean_data/AAPL_stock.csv — check your path.")

