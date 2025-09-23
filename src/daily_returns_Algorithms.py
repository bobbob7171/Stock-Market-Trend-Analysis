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
   try:
     # Load dataset
     df = pd.read_csv("Clean_data/AAPL_stock.csv", parse_dates=["Date"])
     df = df.sort_values("Date").dropna(subset=["Close"]).reset_index(drop=True)

     # Extract close prices
     prices = df["Close"].astype(float).tolist()

     d_returns = daily_returns(prices)

     dates = df["Date"].dt.strftime("%Y-%m-%d").tolist()

     max_days = len(prices) - 1

     days_count = int(input(f"Enter number of days to display (max {max_days}): "))

     if days_count < 1 or days_count > max_days:
         raise ValueError(f"Days count must be between 1 and {max_days}.")
     print("\n===== Apple Stock Daily Returns Analysis =====")
     print(f"First {days_count} Daily Returns:")
     print("---------------------------------------------")
     for i in range(1, days_count + 1):
         print(f"Date: {dates[i]} Price: ${prices[i]:,.2f} Daily Return: {d_returns[i]:.4%}")

   except FileNotFoundError:
      print("Could not find Clean_data/AAPL_stock.csv — check your path.")

