# main.py

from data_handler import (
    CONFIG, fetch_stock_data, clean_all_tickers, align_with_trading_days,
    detect_outliers, data_quality_report, validate_preprocessing,
    pivot_wide, save_datasets
)

def main():
    # 1️.Fetch Data
    raw_data = fetch_stock_data(
        tickers=CONFIG["tickers"],
        start=CONFIG["start_date"],
        end=CONFIG["end_date"],
        backup_path=CONFIG["backup_path"]
    )

    # 2.Clean Data
    cleaned_data = clean_all_tickers(raw_data, CONFIG["tickers"])

    # 3.Align with Trading Days
    cleaned_data = align_with_trading_days(cleaned_data, CONFIG["start_date"], CONFIG["end_date"])

    # 4️.Detect Outliers
    cleaned_data = detect_outliers(cleaned_data)

    # 5️.Validate Preprocessing
    validate_preprocessing(cleaned_data, CONFIG["start_date"], CONFIG["end_date"])

    # 6.Generate Data Quality Report
    quality_report = data_quality_report(cleaned_data)
    print("📊 Data Quality Report")
    print(quality_report)

    # 7️. Pivot to Wide Format
    pivot_data = pivot_wide(cleaned_data)
    print(f"📊 Wide-format data shape: {pivot_data.shape}")

    # # 8️. Optional: Plot Outliers (example for a single ticker)
    # plot_outliers(cleaned_data, "AAPL")
    # # plot_outliers(cleaned_data, ["AAPL", "MSFT"])  # Example for multiple tickers

    # 9️. Save Cleaned Datasets
    save_datasets(cleaned_data, pivot_data, CONFIG)

    print("✅ All steps completed successfully!")


if __name__ == "__main__":
    main()
