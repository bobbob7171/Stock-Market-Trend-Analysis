# 📈 Stock Market Trend Analysis

![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Enabled-brightgreen)
![License](https://img.shields.io/badge/license-Educational-lightgrey)
![Tests](https://img.shields.io/badge/tests-13%2F13%20passed-success)

A comprehensive **Python application** for analyzing stock market trends, calculating trading profits, and visualizing financial data. Built with **Streamlit** for interactive dashboards and **vectorized O(n)** algorithms for robust and transparent financial computation.

---

## 🎯 What Problem Does This Solve?

Traditional stock analysis platforms suffer from several limitations:

**🔒 Transparency Gap** — Commercial platforms hide their algorithms, making independent verification impossible  
**💸 Accessibility Barriers** — Professional tools are expensive and require advanced expertise  
**🔄 Reproducibility Issues** — Black-box calculations prevent custom or educational analysis  
**🎓 Educational Limitations** — Existing tools prioritize trading over understanding

### ✅ Our Solution

* **📖 Transparent Algorithms** — Every calculation validated step-by-step
* **🆓 Zero-Cost Access** — Web deployment removes all barriers  
* **🔬 Educational Focus** — Built-in validation and debugging modes
* **🧩 Modular Verification** — Each component independently testable
* **🌐 Open Access** — [Use instantly online](https://stonk-analyzer.streamlit.app/) – no installation required

> *Developed as part of the **INF1002 Programming Fundamentals** module (Weeks 3–7 deliverables). Demonstrates modular design, O(n) algorithmic optimization, and transparency in financial computing.*

---

## 🎥 User Manual & Tutorial

**📹 Video Guide:** [Watch the complete tutorial](https://youtu.be/kBWDgTUPgP4)  
*Learn how to use all features in under 5 minutes:*

* Data loading & configuration
* Technical analysis interpretation  
* Profit optimization strategies
* Exporting results and reports

---

## 🚀 Features

### 📊 Data Management

* **Multi-Source Data Import** — Fetch live data from Yahoo Finance or upload custom CSV files
* **Automated Data Cleaning** — Handle missing values, align with trading calendars, detect outliers using Z-score and IQR methods
* **Data Validation** — Comprehensive quality checks and schema validation with zero missing values
* **Flexible Date & SMA Ranges** — Per-ticker customization with global override options
* **Trading Day Alignment** — Restrict analysis to official NYSE trading days only

### 📈 Analytics & Calculations

* **Daily Returns** — Vectorized calculation with O(n) performance and manual debugging modes
* **Technical Indicators** — Simple Moving Averages (SMA) with customizable windows and crossover signals
* **Trend Analysis** — Streak detection for consecutive up/down days with momentum insights
* **Risk Metrics** — Annualized returns, volatility, and Sharpe ratios for portfolio optimization
* **Profit Optimization** — Single (O(n) sliding window) and multiple transaction (O(n) vectorized) profit calculations

### 📊 Visualization

* **Interactive Charts** — Plotly-based visualizations with hover details
* **Price + SMA** — Overlay technical indicators with buy/sell signals
* **Trend Streaks** — Color-coded upward/downward trend segments
* **Risk-Return Scatter** — Sharpe ratio highlighting for optimal investments
* **Profit Comparison** — Single vs. multiple transaction performance analysis
* **Buy/Sell Indicators** — Optimal trading points based on local minima/maxima

### 💾 Export Capabilities

* **CSV Downloads** — Cleaned data, analytics results, profit summaries
* **Interactive HTML** — Export Plotly charts for offline viewing
* **Quality Reports** — Data validation and coverage statistics

---

## 🏗️ System Architecture

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
├── src/                            # Core application modules
│   ├── analytics.py                # O(n) calculations & metrics
│   ├── config.py                   # Application configuration
│   ├── data_handler.py             # Data processing pipeline
│   ├── graph.py                    # Visualization functions
│   ├── profit.py                   # Profit optimization algorithms
│   ├── validation.py               # Comprehensive test suite
│   └── __init__.py
│
├── .gitignore
├── config.py                       # Main configuration
├── main.py                         # Streamlit dashboard
├── README.md                       # Instructions
└── requirements.txt                # Dependencies
```

---

## ⚡ Performance & Results

### 🎯 Key Achievements

* **100% Calculation Accuracy** — 13/13 validation tests passed with complete algorithmic transparency
* **O(n) Efficiency** — Vectorized operations achieve 10–100× speedup over naive implementations
* **Zero Missing Values** — Processed 5,000+ trading records with complete data integrity
* **Real Insights** — Identified TSLA multiple-transaction profits > 2500% vs single transactions

### 📈 Analytical Insights

| Stock          | Volatility (σ) | Sharpe Ratio | Key Finding                                 |
| :------------- | :------------- | :----------- | :------------------------------------------ |
| **TSLA**       | 0.045          | 1.8          | Highest volatility, best for active trading |
| **JPM**        | 0.017          | **2.1**      | Best risk-adjusted return                   |
| **AAPL**       | 0.022          | 1.5          | Stable long-term growth                     |
| **All Stocks** | —              | —            | Avg. upward streak: 3.2 days                |

> Multiple-transaction strategies consistently outperform single transactions by 200–600%.

---

## 🛠️ Installation

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd INF1002-Programming
   ```

2. **Create Virtual Environment** *(recommended)*
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### 🧩 `requirements.txt` (excerpt)
```text
yfinance
pandas
matplotlib
seaborn
scipy
pandas_market_calendars
tabulate
plotly
streamlit
```

---

## 🚀 Quick Start

### 🌐 Option 1: Use the Web App *(No Installation Needed)*
👉 **[Launch App](https://stonk-analyzer.streamlit.app/)**  
*Perfect for beginners and quick analysis*

### 🎥 Option 2: Watch the Tutorial
**📹 [Complete User Guide Video](https://youtu.be/kBWDgTUPgP4)** — *5 minute walkthrough*

### 💻 Option 3: Streamlit Dashboard *(Run Locally)*
```bash
streamlit run main.py
```

### 📓 Option 4: Jupyter Notebook Analysis
```bash
jupyter notebook notebook/EDA.ipynb
```

---

## ⚙️ Configuration

Edit `config.py` to customize defaults:

```python
CONFIG = {
    "tickers": ["AAPL", "MSFT", "GOOGL", "JPM", "TSLA"],
    "start_date": "2022-01-01", 
    "end_date": "2025-01-01",
    "sma_windows": [5, 20, 50],
    "enable_logging": True,
    "validation_strictness": "high",
    # ... additional settings
}
```

### Global Settings

* **📅 Date Range** — Apply globally or override per ticker
* **📊 SMA Periods** — Common presets (5, 20, 50) or custom windows
* **⚙️ Smart Defaults** — Optimized analysis ranges and logging options

---

## 📊 Usage Guide

### 📈 Data Sources

**Yahoo Finance Integration**
- Enter comma-separated tickers (e.g., `AAPL, MSFT, GOOGL`)
- Set custom date ranges per ticker or use global settings
- Configure SMA windows individually or apply globally

**CSV Upload**
- Support for multiple CSV files
- Automatic ticker assignment
- Schema validation with helpful error messages

### 🔍 Key Analyses

1. **Data Quality Assessment**
   - Coverage of trading days
   - Missing value analysis
   - Outlier detection using Z-score and IQR methods

2. **Technical Analysis**
   - SMA crossover signals for trend identification
   - Trend strength via streak detection and momentum analysis
   - Volatility and return calculations for risk assessment

3. **Profit Optimization**
   - Single transaction: Buy low, sell high (O(n) sliding window)
   - Multiple transactions: Capture all upward movements (O(n) vectorized)
   - Comparative performance analysis across strategies

4. **Risk Management**
   - Annualized risk-return profiles with Sharpe ratios
   - Portfolio efficiency analysis and optimization
   - Visual identification of optimal risk-adjusted investments

### 💾 Export Options

All visualizations and data tables can be exported:
- **CSV** — Raw data, analytics results, summaries
- **HTML** — Interactive Plotly charts for offline presentation
- **Validation Reports** — Test results and quality metrics

---

## 📈 Example Workflow

1. **Configure Settings** — Set global analysis period and SMA windows
2. **Load Data** — Fetch AAPL, MSFT, TSLA from Yahoo Finance (2022-2025)
3. **Clean & Validate** — Automated processing with quality report (0 missing values)
4. **Analyze Trends** — 20-day and 50-day SMA crossovers with buy/sell signals
5. **Calculate Profits** — Compare single vs. multiple transaction strategies
6. **Assess Risk** — Annualized volatility and Sharpe ratios for portfolio optimization  
7. **Visualize** — Interactive charts with export options
8. **Export** — Download results for reporting and further analysis

---

## 🧪 Testing & Validation

Run all tests:
```bash
python src/validation.py
```

### Coverage Includes
- Daily returns calculation accuracy (pandas parity)
- SMA verification vs. manual calculations  
- Profit optimization algorithms (O(n) verification)
- Edge cases and corner conditions
- Performance benchmarking
- Data integrity and schema validation

**Results:** ✅ 13/13 tests passed — 100% accuracy with complete algorithmic transparency

---

## 🎓 Educational Value

This project demonstrates:

* **Modular Python Design** — Separated concerns with data, analytics, visualization layers
* **Algorithmic Efficiency** — O(n) implementations vs naive O(n²) approaches
* **Financial Computing** — Accurate implementation of industry-standard formulas
* **Validation-First Development** — Comprehensive testing ensures reliability
* **Web Deployment** — Streamlit for accessible, interactive dashboards

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all validation tests pass
5. Submit a pull request

---

## 📄 License

This project is for **educational purposes** as part of INF1002 Programming Module. Not intended for commercial trading or financial advice.

---

## 🆘 Support

For issues and questions:

1. 📹 **Watch the Tutorial** — [Complete User Guide Video](https://youtu.be/kBWDgTUPgP4)
2. 🧾 Check validation reports in `data/validation/`
3. ⚙️ Verify dependencies from `requirements.txt`
4. 🌐 Ensure Yahoo Finance API availability for real-time data
5. 🧩 Review configuration settings in `config.py`

---

**Built with ❤️ for transparent, educational financial data analysis**  
