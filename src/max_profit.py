import pandas as pd

def max_profit(prices): 
        # --- Input Validation ---
    if not isinstance(prices, list):
        raise TypeError("Input must be a list.")
    if not all(isinstance(price, (int, float)) for price in prices):
        raise TypeError("All elements must be numeric.")
            
        # --- Profit Calculation ---
    total_profit = sum(max(prices[i] - prices[i - 1], 0) for i in range(1, len(prices)))
    return total_profit



def read_prices_from_csv(file_path, column_name='Close'):
    """
    Reads a specified column from a CSV file into a list.

    Args:
        file_path (str): The path to the CSV file.
        column_name (str): The name of the column containing the prices.

    Returns:
        list: A list of floating-point numbers representing the prices.
              Returns an empty list if an error occurs.
    """
    print(f"\nReading data from '{file_path}'...")
    try:
        data = pd.read_csv(file_path)
        if data.empty or column_name not in data.columns:
            return []
        
        return data[column_name].tolist()
    
    except Exception as e:
        return []

def run_csv_demonstration():
    """
    Demonstrates the max_profit function using data from a local CSV file.
    """
    print("\n--- Running Local CSV Data Demonstration ---")
    
    csv_file = 'stock_data.csv'  
    
    try:
        prices = read_prices_from_csv(csv_file)
        profit = max_profit(prices)
        print(f"\nMaximum Profit based on data from '{csv_file}': ${profit:.2f}")
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found.")
    except ValueError as e:
        print(f"Error: {e}")
    except TypeError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")



# --- Main Execution Block ---
if __name__ == "__main__":
    run_csv_demonstration()
