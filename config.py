CONFIG = {
    # ------------------------
    # Tickers and Date Range
    # ------------------------
    "tickers": ["AAPL", "MSFT", "GOOGL", "JPM", "TSLA"],
    "start_date": "2022-01-01",
    "end_date": "2025-01-01",

    # ------------------------
    # File Paths
    # ------------------------
    "backup_path": "../data/backup_stocks.csv",
    "output_long": "../data/cleaned_stock_data_long.csv",
    "output_wide": "../data/cleaned_stock_data_wide.csv",

    # ------------------------
    # Analytics Output Paths
    # ------------------------
    "analytics_path": "../data/analytics/analytics_stock_data.csv",
    "returns_wide_path": "../data/analytics/daily_returns_wide.csv",
    "streaks_path": "../data/analytics/streaks_for_plot.csv",
    "streak_summary_path": "../data/analytics/streak_summary.csv",
    "profit_summary_path": "../data/analytics/profit_summary.csv",

    # ------------------------
    # Analysis Settings
    # ------------------------
    "sma_windows": [5, 20, 50],
    "enable_logging": True,

    # ------------------------
    # Toggle Debug Columns
    # ------------------------
    "keep_debug": False,   # If False → drops manual/vectorized debug cols
}
