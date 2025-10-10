# 📈 Stock Market Trend Analysis

A comprehensive Python application for analyzing stock market trends, calculating trading profits, and visualizing financial data. Built with Streamlit for interactive dashboarding and robust data processing capabilities.

## 🚀 Features

### Data Management
- **Multi-source Data Import**: Fetch real-time data from Yahoo Finance or upload custom CSV files
- **Automated Data Cleaning**: Handle missing values, align with trading calendars, detect outliers
- **Data Validation**: Comprehensive quality checks and schema validation
- **Flexible Date Ranges**: Per-ticker date customization with global override options

### Analytics & Calculations
- **Daily Returns**: Vectorized calculation with manual debugging options
- **Technical Indicators**: Simple Moving Averages (SMA) with customizable windows
- **Trend Analysis**: Streak detection for consecutive up/down days
- **Risk Metrics**: Annualized returns, volatility, and Sharpe ratios
- **Profit Optimization**: Single and multiple transaction profit calculations

### Visualization
- **Interactive Charts**: Plotly-based visualizations with hover details
- **Price + SMA**: Overlay technical indicators with buy/sell signals
- **Trend Streaks**: Color-coded upward/downward trend segments
- **Risk-Return Scatter**: Sharpe ratio highlighting for optimal investments
- **Profit Comparison**: Single vs. multiple transaction performance

### Export Capabilities
- **CSV Downloads**: Cleaned data, analytics results, profit summaries
- **Interactive HTML**: Export Plotly charts for offline viewing
- **Quality Reports**: Data validation and coverage statistics

## 📁 Project Structure

```
INF1002-Programming/
│
├── data/                           # Processed data outputs
│   ├── cleaned_stock_data_long.csv
│   ├── cleaned_stock_data_wide.csv
│   ├── analytics/                  # Analytics outputs
│   │   ├── analytics_stock_data.csv
│   │   ├── daily_returns_wide.csv
│   │   ├── profit_summary.csv
│   │   ├── streaks_for_plot.csv
│   │   └── streak_summary.csv
│   └── validation/                 # Test results
│       └── validation_results.txt
│
├── notebook/                       # Exploratory analysis
│   └── EDA.ipynb
│
├── src/                           # Core application modules
│   ├── analytics.py              # Calculations & metrics
│   ├── config.py                 # Application configuration
│   ├── data_handler.py           # Data processing pipeline
│   ├── graph.py                  # Visualization functions
│   ├── profit.py                 # Profit optimization
│   ├── validation.py             # Test suite
│   └── __init__.py
│
├── .gitignore
├── config.py                      # Main configuration
├── main.py                        # Streamlit dashboard
└── requirements.txt              # Dependencies
```

## 🛠 Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd INF1002-Programming
   ```

2. **Create virtual environment** *(recommended)*

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Quick Start

### 🌐 Option 1: Use the Web App *(No Installation Needed)*

👉 [Launch App](https://stonk-analyzer.streamlit.app/)

### 💻 Option 2: Streamlit Dashboard *(Run Locally)*

```bash
streamlit run main.py
```

### 📓 Option 3: Jupyter Notebook Analysis

```bash
jupyter notebook notebook/EDA.ipynb
```

### 🧪 Option 4: Run Validation Tests

```bash
python src/validation.py
```

## 📊 Usage Guide

### Data Sources

**Yahoo Finance Integration**
- Enter comma-separated tickers (e.g., `AAPL, MSFT, GOOGL`)
- Set custom date ranges per ticker
- Configure SMA windows individually

**CSV Upload**
- Support for multiple CSV files
- Automatic ticker assignment
- Schema validation with helpful error messages

### Key Analyses

1. **Data Quality Assessment**
   - Coverage of trading days
   - Missing value analysis
   - Outlier detection using Z-score and IQR methods

2. **Technical Analysis**
   - SMA crossover signals
   - Trend strength via streak detection
   - Volatility and return calculations

3. **Profit Optimization**
   - Single transaction: Buy low, sell high
   - Multiple transactions: Capture all upward movements
   - Comparative performance analysis

4. **Risk Management**
   - Annualized risk-return profiles
   - Sharpe ratio for risk-adjusted returns
   - Portfolio efficiency analysis

### Export Options

All visualizations and data tables can be exported:
- **CSV**: Raw data, analytics results, summaries
- **HTML**: Interactive Plotly charts
- **Validation Reports**: Test results and quality metrics

## 🔧 Configuration

Edit `config.py` to customize:

```python
CONFIG = {
    "tickers": ["AAPL", "MSFT", "GOOGL", "JPM", "TSLA"],
    "start_date": "2022-01-01",
    "end_date": "2025-01-01",
    "sma_windows": [5, 20, 50],
    "enable_logging": True,
    # ... additional settings
}
```

## 📈 Example Workflow

1. **Load Data**: Fetch AAPL, MSFT, TSLA from 2022-2025
2. **Clean & Validate**: Automated processing with quality report
3. **Analyze Trends**: 20-day and 50-day SMA crossovers
4. **Calculate Profits**: Compare single vs. multiple transaction strategies
5. **Assess Risk**: Annualized volatility and Sharpe ratios
6. **Visualize**: Interactive charts with export options
7. **Export**: Download results for reporting

## 🧪 Testing & Validation

The application includes comprehensive tests:

```bash
python src/validation.py
```

Tests cover:
- Daily returns calculation accuracy
- SMA computation against manual calculations
- Profit optimization algorithms
- Edge cases and corner conditions
- Performance benchmarking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all validation tests pass
5. Submit a pull request

## 📄 License

This project is for educational purposes as part of INF1002 Programming course.

## 🆘 Support

For issues and questions:
1. Check the validation test results
2. Review data quality reports
3. Ensure all dependencies are installed
4. Verify Yahoo Finance API availability for real-time data

---

**Built with ❤️ for financial data analysis and algorithmic trading education**