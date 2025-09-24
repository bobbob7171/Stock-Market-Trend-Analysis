import math
import pandas as pd
from pathlib import Path
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
        # Load dataset
        df = pd.read_csv('data/cleaned_stock_data_wide.csv')
        date_column = df.columns[0]
        stocks = list(df.columns[1:])

        #Ask user to select stock
        print("Available stocks:",", ".join(stocks))
        stock_choice = input("Enter stock symbol: ").strip().upper()

        if stock_choice not in stocks:
            raise ValueError(f"Stock '{stock_choice}' not found in dataset.")

        # Extract close prices
        prices = df[stock_choice].tolist()

try:
        # Ask user for window size
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

        print("\n===== Stock Analysis =====")
        print(f"File: {stock_choice}")
        print(f"First {num_closing} Closing Prices:")
        print("------------------------------")
        for i, price in enumerate(prices[:num_closing], 1):
            print(f"{i:2d}: ${price:,.2f}")

        print(f"\nFirst {count} SMA({window_size}) Values:")
        print("------------------------------")
        for i in range(start, stop):
            print(f"Day: {i+1:3d}: ${sma_cal[i]:,.2f}")
        print("===============================\n")

        # build a plotting-ready DataFrame (Date, Close ,SMA)
        # Select only date and chosen stock columns
        # Rename Date columns to Close columns so file is consistent
        plot_df = df[[date_column, stock_choice]].rename(
            columns={date_column: "Date", stock_choice: "Close"}
            )
        plot_df[f"SMA({window_size})"] = sma_cal # add SMA column

        # Saves the output to out file
        out_dir = Path("out"); out_dir.mkdir(exist_ok=True, parents=True)
        out_path = out_dir / f"{stock_choice}_for_plot.csv"
        plot_df.to_csv(out_path, index=False)


         # preview for sanity-check
        print("Preview of file written for plotting:")
        print(plot_df.tail(10).to_string(index=False))     

except FileNotFoundError:
    print("Could not find data. Please check file path.")

    # Compute SMA using Pandas rolling mean
sma_pd = pd.Series(prices).rolling(window_size).mean().tolist()

# Loops through both my SMAs (sma_cal) and Pandas SMAs (sma_pd) for comparison
for i, (smacal, pandascal) in enumerate(zip(sma_cal, sma_pd)):
     if smacal is None and pd.isna(pandascal): # if both are None/NaN means they both have no value, they match
        continue  # skip to next data point
     elif smacal is not None and not pd.isna(pandascal):# if both has values (not empty)
          if not math.isclose(smacal, pandascal, rel_tol=1e-12): # compare values with tolerance(makes sure it doesnt call mismatch if numbers are diff by a very small amount)
               print(f"Mismatch at index {i}: mine={smacal}, pandas={pandascal}")
               break
else: # if no mismatch found prints the following
     print("SMA values are correct!")
    