"""
validation.py - Validation Test Suite
Tests analytics.py and profit.py implementations against trusted sources and manual calculations.
"""

import pandas as pd
import numpy as np
import sys
import os
from io import StringIO

from typing import Dict, List, Tuple

# Mock CONFIG for testing
class MockConfig:
    def get(self, key, default=None):
        configs = {
            "keep_debug": False,
            "enable_logging": True,
            "sma_windows": [20, 50],
            "tickers": ["AAPL", "GOOGL", "MSFT"]
        }
        return configs.get(key, default)

# Replace config in analytics and profit modules
sys.modules['config'] = type(sys)('config')
sys.modules['config'].CONFIG = MockConfig()

# Import after config mock
from analytics import calculate_daily_returns, calculate_sma, detect_streaks, calculate_annual_risk_return
from profit import max_profit_single, max_profit_multiple


# ============================================================================
# OUTPUT CAPTURE UTILITY
# ============================================================================

class OutputCapture:
    """Captures both console output and stores it for file writing."""
    def __init__(self):
        self.buffer = StringIO()
        self.original_stdout = sys.stdout
        
    def start(self):
        """Start capturing output."""
        sys.stdout = self
        
    def stop(self):
        """Stop capturing output."""
        sys.stdout = self.original_stdout
        
    def write(self, text):
        """Write to both console and buffer."""
        self.original_stdout.write(text)
        self.buffer.write(text)
        
    def flush(self):
        """Flush both outputs."""
        self.original_stdout.flush()
        
    def get_output(self):
        """Get captured output as string."""
        return self.buffer.getvalue()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_test_data(prices: List[float], ticker: str = "TEST") -> pd.DataFrame:
    """Create a simple test DataFrame with given prices."""
    dates = pd.date_range(start="2024-01-01", periods=len(prices), freq="D")
    return pd.DataFrame({
        "Date": dates,
        "Ticker": ticker,
        "Close": prices
    })


def manual_daily_return(prices: List[float]) -> List[float]:
    """Manually calculate daily returns."""
    returns = [np.nan]  # First day has no return
    for i in range(1, len(prices)):
        if prices[i-1] != 0:
            returns.append((prices[i] - prices[i-1]) / prices[i-1])
        else:
            returns.append(np.nan)
    return returns


def manual_sma(prices: List[float], window: int) -> List[float]:
    """Manually calculate SMA."""
    sma = []
    for i in range(len(prices)):
        if i < window - 1:
            sma.append(np.nan)
        else:
            sma.append(np.mean(prices[i - window + 1 : i + 1]))
    return sma


def are_close(a, b, tolerance=1e-9) -> bool:
    """Compare two values/arrays with tolerance for floating point errors."""
    if isinstance(a, (list, np.ndarray, pd.Series)):
        a_arr = np.array(a, dtype=float)
        b_arr = np.array(b, dtype=float)
        # Handle NaN comparisons
        mask_nan = np.isnan(a_arr) & np.isnan(b_arr)
        mask_valid = ~np.isnan(a_arr) & ~np.isnan(b_arr)
        if not np.all(mask_nan | mask_valid):
            return False
        return np.allclose(a_arr[mask_valid], b_arr[mask_valid], rtol=tolerance, atol=tolerance)
    else:
        if np.isnan(a) and np.isnan(b):
            return True
        return abs(a - b) < tolerance


# ============================================================================
# TEST CASES
# ============================================================================

def test_case_1_daily_returns_manual():
    """
    TEST CASE 1: Daily Returns - Manual Calculation
    Compare our implementation against hand-calculated returns.
    """
    print("\n" + "="*80)
    print("TEST CASE 1: Daily Returns - Manual Calculation")
    print("="*80)
    
    # Simple test data
    prices = [100, 105, 103, 110, 108]
    df = create_test_data(prices)
    
    # Expected manual calculation:
    # Day 0: NaN (no previous day)
    # Day 1: (105-100)/100 = 0.05
    # Day 2: (103-105)/105 = -0.019047619
    # Day 3: (110-103)/103 = 0.067961165
    # Day 4: (108-110)/110 = -0.018181818
    expected = manual_daily_return(prices)
    
    # Run our implementation
    result_df = calculate_daily_returns(df)
    actual = result_df["Daily_Return"].tolist()
    
    print(f"Prices:          {prices}")
    print(f"Expected returns: {[f'{x:.6f}' if not np.isnan(x) else 'NaN' for x in expected]}")
    print(f"Actual returns:   {[f'{x:.6f}' if not np.isnan(x) else 'NaN' for x in actual]}")
    
    passed = are_close(expected, actual)
    print(f"\n✓ PASSED" if passed else "\n✗ FAILED")
    return passed


def test_case_2_daily_returns_pandas():
    """
    TEST CASE 2: Daily Returns - Compare with pandas pct_change()
    Validate against pandas' built-in function (trusted source).
    """
    print("\n" + "="*80)
    print("TEST CASE 2: Daily Returns - Compare with pandas pct_change()")
    print("="*80)
    
    # More complex data with multiple tickers
    df = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=10, freq="D").tolist() * 2,
        "Ticker": ["AAPL"] * 10 + ["GOOGL"] * 10,
        "Close": [150, 152, 151, 155, 153, 156, 158, 157, 160, 162,
                  2800, 2820, 2810, 2850, 2830, 2860, 2880, 2870, 2900, 2920]
    })
    
    # Trusted pandas calculation
    expected = df.groupby("Ticker")["Close"].pct_change()
    
    # Our implementation
    result_df = calculate_daily_returns(df)
    actual = result_df["Daily_Return"]
    
    print(f"Sample data (first 5 rows):")
    print(df.head())
    print(f"\nExpected (pandas): {expected.values[:5]}")
    print(f"Actual (ours):     {actual.values[:5]}")
    
    passed = are_close(expected.values, actual.values)
    print(f"\n✓ PASSED" if passed else "\n✗ FAILED")
    return passed


def test_case_3_sma_manual():
    """
    TEST CASE 3: SMA - Manual Calculation for Small Window
    Hand-calculate SMA with window=3 and compare.
    """
    print("\n" + "="*80)
    print("TEST CASE 3: SMA - Manual Calculation (window=3)")
    print("="*80)
    
    prices = [10, 12, 14, 13, 15, 16, 14]
    df = create_test_data(prices)
    
    # Manual SMA calculation for window=3:
    # Day 0: NaN (not enough data)
    # Day 1: NaN (not enough data)
    # Day 2: (10+12+14)/3 = 12.0
    # Day 3: (12+14+13)/3 = 13.0
    # Day 4: (14+13+15)/3 = 14.0
    # Day 5: (13+15+16)/3 = 14.666...
    # Day 6: (15+16+14)/3 = 15.0
    expected = manual_sma(prices, window=3)
    
    # Our implementation
    result_df = calculate_sma(df, windows=[3])
    actual = result_df["SMA_3"].tolist()
    
    print(f"Prices:      {prices}")
    print(f"Expected SMA: {[f'{x:.4f}' if not np.isnan(x) else 'NaN' for x in expected]}")
    print(f"Actual SMA:   {[f'{x:.4f}' if not np.isnan(x) else 'NaN' for x in actual]}")
    
    passed = are_close(expected, actual)
    print(f"\n✓ PASSED" if passed else "\n✗ FAILED")
    return passed


def test_case_4_sma_pandas_rolling():
    """
    TEST CASE 4: SMA - Compare with pandas rolling().mean()
    Validate against pandas' built-in rolling window function.
    """
    print("\n" + "="*80)
    print("TEST CASE 4: SMA - Compare with pandas rolling().mean()")
    print("="*80)
    
    # Random prices for more robust testing
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(50))
    df = create_test_data(prices.tolist())
    
    window = 20
    
    # Trusted pandas calculation
    expected = df["Close"].rolling(window=window, min_periods=window).mean()
    
    # Our implementation
    result_df = calculate_sma(df, windows=[window])
    actual = result_df[f"SMA_{window}"]
    
    print(f"Window size: {window}")
    print(f"Data points: {len(prices)}")
    print(f"Expected (first 25): {expected.values[:25]}")
    print(f"Actual (first 25):   {actual.values[:25]}")
    
    passed = are_close(expected.values, actual.values)
    print(f"\n✓ PASSED" if passed else "\n✗ FAILED")
    return passed


def test_case_5_sma_corner_cases():
    """
    TEST CASE 5: SMA - Corner Cases
    Test edge cases: data shorter than window, single value, empty data.
    """
    print("\n" + "="*80)
    print("TEST CASE 5: SMA - Corner Cases")
    print("="*80)
    
    all_passed = True
    
    # Corner case 1: Data shorter than window
    print("\nCorner Case 5a: Data shorter than SMA window")
    prices_short = [100, 102, 101]
    df_short = create_test_data(prices_short)
    result = calculate_sma(df_short, windows=[5])
    # All values should be NaN
    expected_all_nan = [np.nan, np.nan, np.nan]
    actual = result["SMA_5"].tolist()
    passed_5a = are_close(expected_all_nan, actual)
    print(f"Prices: {prices_short}, Window: 5")
    print(f"Expected: All NaN")
    print(f"Actual: {actual}")
    print(f"✓ PASSED" if passed_5a else "✗ FAILED")
    all_passed = all_passed and passed_5a
    
    # Corner case 2: Window size = data length
    print("\nCorner Case 5b: Window size equals data length")
    prices_exact = [100, 105, 103, 110, 108]
    df_exact = create_test_data(prices_exact)
    result = calculate_sma(df_exact, windows=[5])
    expected_last = np.mean(prices_exact)
    actual = result["SMA_5"].tolist()
    passed_5b = np.isnan(actual[:-1]).all() and are_close(actual[-1], expected_last)
    print(f"Prices: {prices_exact}, Window: 5")
    print(f"Expected: [NaN, NaN, NaN, NaN, {expected_last}]")
    print(f"Actual: {actual}")
    print(f"✓ PASSED" if passed_5b else "✗ FAILED")
    all_passed = all_passed and passed_5b
    
    # Corner case 3: Window = 1 (each value is its own SMA)
    print("\nCorner Case 5c: Window size = 1")
    prices_one = [100, 105, 103]
    df_one = create_test_data(prices_one)
    result = calculate_sma(df_one, windows=[1])
    expected = prices_one
    actual = result["SMA_1"].tolist()
    passed_5c = are_close(expected, actual)
    print(f"Prices: {prices_one}, Window: 1")
    print(f"Expected: {expected}")
    print(f"Actual: {actual}")
    print(f"✓ PASSED" if passed_5c else "✗ FAILED")
    all_passed = all_passed and passed_5c
    
    print(f"\n{'✓ ALL CORNER CASES PASSED' if all_passed else '✗ SOME CORNER CASES FAILED'}")
    return all_passed


def test_case_6_streaks_manual():
    """
    TEST CASE 6: Streak Detection - Manual Calculation
    Hand-calculate streaks for a simple returns pattern.
    """
    print("\n" + "="*80)
    print("TEST CASE 6: Streak Detection - Manual Calculation")
    print("="*80)
    
    # Create data with known return pattern
    # Pattern: up, up, down, down, down, up, zero, up
    prices = [100, 105, 110, 108, 105, 102, 106, 106, 110]
    df = create_test_data(prices)
    df = calculate_daily_returns(df)
    
    # Manual streak calculation:
    # Return: [NaN,  +,   +,   -,   -,   -,   +,   0,   +]
    # Streak: [NaN,  1,   2,  -1,  -2,  -3,   1,   0,   1]
    expected_streaks = [np.nan, 1, 2, -1, -2, -3, 1, 0, 1]
    
    result_df = detect_streaks(df)
    actual_streaks = result_df["Streak"].tolist()
    
    returns = df["Daily_Return"].tolist()
    print(f"Prices:  {prices}")
    print(f"Returns: {[f'{x:.4f}' if not np.isnan(x) else 'NaN' for x in returns]}")
    print(f"Expected Streaks: {expected_streaks}")
    print(f"Actual Streaks:   {actual_streaks}")
    
    passed = are_close(expected_streaks, actual_streaks)
    print(f"\n✓ PASSED" if passed else "\n✗ FAILED")
    return passed

def test_case_6b_streaks_corner_cases():
    """
    TEST CASE 6b: Streak Detection - Corner Cases
    Tests single-day, constant, and zero-return scenarios.
    """
    print("\n" + "="*80)
    print("TEST CASE 6b: Streak Detection - Corner Cases")
    print("="*80)
    
    all_passed = True
    
    # Single-day data
    df_single = create_test_data([100])
    df_single = calculate_daily_returns(df_single)
    result = detect_streaks(df_single)
    passed_single = np.isnan(result["Streak"].iloc[0])
    print(f"Single-day streak: {result['Streak'].tolist()} -> {'PASS' if passed_single else 'FAIL'}")
    all_passed &= passed_single
    
    # Constant prices
    df_const = create_test_data([100, 100, 100, 100])
    df_const = calculate_daily_returns(df_const)
    result = detect_streaks(df_const)
    expected = [np.nan, 0, 0, 0]  # streak 0 for no change
    passed_const = are_close(result["Streak"].tolist(), expected)
    print(f"Constant streak: {result['Streak'].tolist()} -> {'PASS' if passed_const else 'FAIL'}")
    all_passed &= passed_const
    
    # Zero returns
    df_zero = create_test_data([100, 100, 100])
    df_zero = calculate_daily_returns(df_zero)
    result = detect_streaks(df_zero)
    expected_zero = [np.nan, 0, 0]
    passed_zero = are_close(result["Streak"].tolist(), expected_zero)
    print(f"Zero-return streak: {result['Streak'].tolist()} -> {'PASS' if passed_zero else 'FAIL'}")
    all_passed &= passed_zero
    
    print(f"\n{'✓ ALL STREAK CORNER CASES PASSED' if all_passed else '✗ SOME FAILED'}")
    return all_passed


def test_case_7_profit_single_manual():
    """
    TEST CASE 7: Max Profit (Single Transaction) - Manual Calculation
    Hand-calculate maximum profit for buy-once-sell-once scenario.
    """
    print("\n" + "="*80)
    print("TEST CASE 7: Max Profit Single Transaction - Manual Calculation")
    print("="*80)
    
    # Test case: Buy at 100, sell at 120 gives profit of 20
    prices = [100, 120, 90, 110, 85, 130, 95]
    df = create_test_data(prices)
    
    # Manual analysis:
    # Best buy: day 4 (85), best sell: day 5 (130)
    # Max profit: 130 - 85 = 45
    expected_profit = 45.0
    
    actual_profit = max_profit_single(df, "TEST")
    
    print(f"Prices: {prices}")
    print(f"Best strategy: Buy at {min(prices)} (index {prices.index(min(prices))}), "
          f"Sell at {max(prices[prices.index(min(prices)):] or [min(prices)])} later")
    print(f"Expected max profit: {expected_profit}")
    print(f"Actual max profit:   {actual_profit}")
    
    passed = are_close(expected_profit, actual_profit)
    print(f"\n✓ PASSED" if passed else "\n✗ FAILED")
    return passed

def test_case_7b_profit_single_manual_robust():
    """
    TEST CASE 7b: Max Profit (Single Transaction) - Robust Check
    Ensure correct buy/sell even with multiple local minima/maxima.
    """
    print("\n" + "="*80)
    print("TEST CASE 7b: Max Profit Single Transaction - Robust Check")
    print("="*80)
    
    prices = [100, 120, 90, 110, 85, 130, 95]
    df = create_test_data(prices)
    
    expected_profit = 45.0  # manually computed: buy at 85, sell at 130
    expected_buy_price = 85.0
    expected_sell_price = 130.0
    
    # Use function with indices - returns 5 values!
    actual_profit, buy_price, sell_price, buy_date, sell_date = max_profit_single(df, "TEST", return_indices=True)
    
    print(f"Prices: {prices}")
    print(f"Expected profit: {expected_profit} (Buy: {expected_buy_price}, Sell: {expected_sell_price})")
    print(f"Actual profit:   {actual_profit} (Buy: {buy_price}, Sell: {sell_price})")
    print(f"Buy date: {buy_date}, Sell date: {sell_date}")
    
    passed = (are_close(expected_profit, actual_profit) and 
              are_close(expected_buy_price, buy_price) and 
              are_close(expected_sell_price, sell_price))
    
    print(f"\n✓ PASSED" if passed else "\n✗ FAILED")
    return passed

def test_case_8_profit_multiple_manual():
    """
    TEST CASE 8: Max Profit (Multiple Transactions) - Manual Calculation
    Hand-calculate profit with unlimited transactions.
    """
    print("\n" + "="*80)
    print("TEST CASE 8: Max Profit Multiple Transactions - Manual Calculation")
    print("="*80)
    
    # Test case: Capture all upward movements
    prices = [100, 120, 90, 110, 85, 130, 95, 105]
    df = create_test_data(prices)
    
    # Manual calculation: sum all positive differences
    # 100→120: +20
    # 120→90: skip (negative)
    # 90→110: +20
    # 110→85: skip (negative)
    # 85→130: +45
    # 130→95: skip (negative)
    # 95→105: +10
    # Total: 20 + 20 + 45 + 10 = 95
    expected_profit = 95.0
    
    actual_profit = max_profit_multiple(df, "TEST")
    
    print(f"Prices: {prices}")
    print(f"Profitable moves:")
    for i in range(1, len(prices)):
        diff = prices[i] - prices[i-1]
        if diff > 0:
            print(f"  {prices[i-1]} → {prices[i]}: +{diff}")
    print(f"Expected total profit: {expected_profit}")
    print(f"Actual total profit:   {actual_profit}")
    
    passed = are_close(expected_profit, actual_profit)
    print(f"\n✓ PASSED" if passed else "\n✗ FAILED")
    return passed


def test_case_9_profit_corner_cases():
    """
    TEST CASE 9: Profit Calculations - Corner Cases
    Test edge cases: constant prices, decreasing prices, single price.
    """
    print("\n" + "="*80)
    print("TEST CASE 9: Profit Calculations - Corner Cases")
    print("="*80)
    
    all_passed = True
    
    # Corner case 1: All prices constant (no profit possible)
    print("\nCorner Case 9a: Constant prices")
    prices_const = [100, 100, 100, 100]
    df_const = create_test_data(prices_const)
    single_profit = max_profit_single(df_const, "TEST")
    multi_profit = max_profit_multiple(df_const, "TEST")
    passed_9a = are_close(single_profit, 0.0) and are_close(multi_profit, 0.0)
    print(f"Prices: {prices_const}")
    print(f"Expected: Single=0.0, Multiple=0.0")
    print(f"Actual: Single={single_profit}, Multiple={multi_profit}")
    print(f"✓ PASSED" if passed_9a else "✗ FAILED")
    all_passed = all_passed and passed_9a
    
    # Corner case 2: Strictly decreasing (no profit possible)
    print("\nCorner Case 9b: Strictly decreasing prices")
    prices_dec = [100, 90, 80, 70, 60]
    df_dec = create_test_data(prices_dec)
    single_profit = max_profit_single(df_dec, "TEST")
    multi_profit = max_profit_multiple(df_dec, "TEST")
    passed_9b = are_close(single_profit, 0.0) and are_close(multi_profit, 0.0)
    print(f"Prices: {prices_dec}")
    print(f"Expected: Single=0.0, Multiple=0.0")
    print(f"Actual: Single={single_profit}, Multiple={multi_profit}")
    print(f"✓ PASSED" if passed_9b else "✗ FAILED")
    all_passed = all_passed and passed_9b
    
    # Corner case 3: Only two prices
    print("\nCorner Case 9c: Only two prices")
    prices_two = [100, 150]
    df_two = create_test_data(prices_two)
    single_profit = max_profit_single(df_two, "TEST")
    multi_profit = max_profit_multiple(df_two, "TEST")
    expected = 50.0
    passed_9c = are_close(single_profit, expected) and are_close(multi_profit, expected)
    print(f"Prices: {prices_two}")
    print(f"Expected: Single=50.0, Multiple=50.0")
    print(f"Actual: Single={single_profit}, Multiple={multi_profit}")
    print(f"✓ PASSED" if passed_9c else "✗ FAILED")
    all_passed = all_passed and passed_9c
    
    print(f"\n{'✓ ALL CORNER CASES PASSED' if all_passed else '✗ SOME CORNER CASES FAILED'}")
    return all_passed


def test_case_10_annual_metrics():
    """
    TEST CASE 10: Annual Risk-Return Metrics - Manual Calculation
    Verify Sharpe ratio calculation against manual computation.
    """
    print("\n" + "="*80)
    print("TEST CASE 10: Annual Risk-Return Metrics - Manual Calculation")
    print("="*80)
    
    # Create simple returns data
    daily_returns = [0.01, -0.005, 0.015, 0.002, -0.01, 0.008, 0.012, -0.003, 0.007, 0.005]
    prices = [100]
    for ret in daily_returns:
        prices.append(prices[-1] * (1 + ret))
    
    df = create_test_data(prices)
    df = calculate_daily_returns(df)
    
    # Manual calculation
    trading_days = 252
    risk_free = 0.03
    
    mean_daily = np.mean(daily_returns)
    std_daily = np.std(daily_returns, ddof=1)  # sample std
    
    annual_return = mean_daily * trading_days
    annual_vol = std_daily * np.sqrt(trading_days)
    sharpe = (annual_return - risk_free) / annual_vol if annual_vol != 0 else 0
    
    # Our implementation
    result = calculate_annual_risk_return(df, risk_free_rate=risk_free)
    
    print(f"Daily returns (sample): {daily_returns[:5]}")
    print(f"\nManual calculation:")
    print(f"  Annual Return: {annual_return:.6f}")
    print(f"  Annual Volatility: {annual_vol:.6f}")
    print(f"  Sharpe Ratio: {sharpe:.6f}")
    print(f"\nOur implementation:")
    print(f"  Annual Return: {result['Annual_Return'].values[0]:.6f}")
    print(f"  Annual Volatility: {result['Annual_Volatility'].values[0]:.6f}")
    print(f"  Sharpe Ratio: {result['Sharpe_Ratio'].values[0]:.6f}")
    
    passed = (
        are_close(result['Annual_Return'].values[0], annual_return, tolerance=1e-6) and
        are_close(result['Annual_Volatility'].values[0], annual_vol, tolerance=1e-6) and
        are_close(result['Sharpe_Ratio'].values[0], sharpe, tolerance=1e-6)
    )
    
    print(f"\n✓ PASSED" if passed else "\n✗ FAILED")
    return passed


def test_case_profit_vectorization():
    """
    TEST CASE 11: Profit Vectorization Check
    Compare naive vs vectorized for single and multiple transactions.
    """
    print("\n" + "="*80)
    print("TEST CASE 11: Profit Vectorization Check")
    print("="*80)
    
    prices = [100, 180, 260, 310, 40, 535, 695]
    df = create_test_data(prices)
    
    # Single transaction naive
    max_naive = 0
    for i in range(len(prices)):
        for j in range(i+1, len(prices)):
            max_naive = max(max_naive, prices[j]-prices[i])
    
    max_vec = max_profit_single(df, "TEST")
    
    # Multiple transaction naive
    multi_naive = sum(max(prices[i+1]-prices[i],0) for i in range(len(prices)-1))
    multi_vec = max_profit_multiple(df, "TEST")
    
    passed_single = are_close(max_naive, max_vec)
    passed_multi = are_close(multi_naive, multi_vec)
    
    print(f"Single Tx -> Naive: {max_naive}, Vectorized: {max_vec}")
    print(f"Multiple Tx -> Naive: {multi_naive}, Vectorized: {multi_vec}")
    print(f"✓ PASSED" if passed_single and passed_multi else "✗ FAILED")
    
    return passed_single and passed_multi


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all validation test cases and report results, saving complete output to a file."""

    # Create output capture
    capture = OutputCapture()
    capture.start()

    print("\n" + "#"*80)
    print("# VALIDATION TEST SUITE")
    print("# Testing analytics.py and profit.py implementations")
    print("#"*80)
    
    tests = [
        ("Daily Returns - Manual", test_case_1_daily_returns_manual),
        ("Daily Returns - Pandas", test_case_2_daily_returns_pandas),
        ("SMA - Manual (window=3)", test_case_3_sma_manual),
        ("SMA - Pandas Rolling", test_case_4_sma_pandas_rolling),
        ("SMA - Corner Cases", test_case_5_sma_corner_cases),
        ("Streaks - Manual", test_case_6_streaks_manual),
        ("Streaks - Corner Cases", test_case_6b_streaks_corner_cases),         
        ("Profit Single - Manual", test_case_7_profit_single_manual),
        ("Profit Single - Robust Check", test_case_7b_profit_single_manual_robust), 
        ("Profit Multiple - Manual", test_case_8_profit_multiple_manual),
        ("Profit - Corner Cases", test_case_9_profit_corner_cases),
        ("Profit - Vectorization Check", test_case_profit_vectorization),       
        ("Annual Metrics - Manual", test_case_10_annual_metrics),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "#"*80)
    print("# TEST SUMMARY")
    print("#"*80)
    total = len(results)
    passed_count = sum(1 for _, p in results if p)
    
    for name, p in results:
        status = "✓ PASS" if p else "✗ FAIL"
        print(f"{status:8} | {name}")
    
    print(f"\n{passed_count}/{total} tests passed ({100*passed_count/total:.1f}%)")
    
    if passed_count == total:
        print("\n🎉 All validation tests passed! Implementations are verified correct.")
    else:
        print(f"\n⚠️  {total - passed_count} test(s) failed. Please review implementations.")

    # Stop capturing and get all output
    capture.stop()
    full_output = capture.get_output()

    # Save complete validation output to a file
    report_file = "../data/validation/validation_results.txt"
    os.makedirs(os.path.dirname(report_file), exist_ok=True)

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(full_output)

    print(f"\nComplete validation report saved to {report_file}")

    return passed_count == total


if __name__ == "__main__":
    run_all_tests()