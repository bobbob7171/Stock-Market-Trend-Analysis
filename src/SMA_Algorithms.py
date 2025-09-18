import pandas as pd
def simple_moving_avg(prices: list[float], window: int) -> list[float | None]:
    if window <1:
        raise ValueError("window must at least 1.")
    n = len(prices)
    if n == 0:
        return[]
    sma = [None] * n
    run_sum = 0.0
    for i, p in enumerate(prices):
        run_sum += p
        if i>= window:
            run_sum -= prices[i - window]
        if i>= window - 1:
            sma[i] = run_sum/window
    return sma

if __name__ == "__main__":
    try:
        # Load dataset
        df = pd.read_csv("Clean_data/AAPL_stock.csv", parse_dates=["Date"])
        df = df.sort_values("Date").dropna(subset=["Close"]).reset_index(drop=True)

        # Extract close prices
        prices = df["Close"].astype(float).tolist()

        # User input window size
        window_size = int(input("Enter the window size: "))

        # Ask user how many values to display
        num_closing = int(input("Enter number of closing prices to display: "))

        if window_size <1:
            raise ValueError("Window must be at least 1.")

        # Calculate SMA using user window
        sma_cal = simple_moving_avg(prices, window_size)

        count = 50

        start = window_size - 1

        stop = min(start+count, len(sma_cal))

        print("\n===== Apple Stock Analysis =====")
        print(f"First {num_closing} Closing Prices:")
        print("------------------------------")
        for i, price in enumerate(prices[:num_closing], 1):
            print(f"{i:2d}: ${price:,.2f}")

        print(f"\nFirst {count} SMA({window_size}) Values:")
        print("------------------------------")
        for i in range(start, stop):
            print(f"Day: {i+1:3d}: ${sma_cal[i]:,.2f}")
        print("===============================\n")

    except FileNotFoundError:
        print("Could not find Clean_data/AAPL_stock.csv — check your path.")