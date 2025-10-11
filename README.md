# 📈 Stock Market Trend Analysis

![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Enabled-brightgreen)
![License](https://img.shields.io/badge/license-Educational-lightgrey)
![Tests](https://img.shields.io/badge/tests-13%2F13%20passed-success)

A comprehensive **Python application** for analyzing stock market trends, calculating trading profits, and visualizing financial data.
Built with **Streamlit** for interactive dashboards and **vectorized O(n)** algorithms for robust and transparent financial computation.

---

## 🎯 What Problem Does This Solve?

Traditional stock analysis platforms exhibit several key limitations:

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

> *Developed as part of the **INF1002 Programming Fundamentals** module (Weeks 3–7 deliverables).
> Demonstrates modular design, O(n) algorithmic optimization, and transparency in financial computing.*

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

* **Multi-Source Data Import:** Fetch live data from Yahoo Finance or upload custom CSV files
* **Automated Cleaning:** Handle missing values, align trading calendars, and detect outliers (Z-score/IQR)
* **Data Validation:** Comprehensive quality checks and schema verification
* **Flexible Date & SMA Ranges:** Per-ticker customization with global overrides
* **Trading Day Alignment:** Restrict analysis to official NYSE trading days

### 📈 Technical Analysis & Calculations

* **Daily Returns:** Vectorized O(n) performance with manual debugging mode
* **SMA Indicators:** Customizable moving-average windows with crossover detection
* **Trend Streaks:** Consecutive up/down streak analysis with momentum insights
* **Risk Metrics:** Annualized returns, volatility, and Sharpe ratios
* **Profit Optimization:** Single (O(n) sliding window) and multiple (O(n) vectorized) transaction strategies

### 📉 Visualization & Insights

* **Interactive Charts:** Plotly-based graphs with hover details
* **Price + SMA Overlay:** Visualize crossovers with buy/sell markers
* **Trend Streaks:** Color-coded segments for upward/downward trends
* **Risk-Return Scatter:** Visual Sharpe ratio analysis for optimal investments
* **Profit Comparison:** Single vs. multiple-transaction performance plots
* **Exportable Visuals:** Download charts as standalone interactive HTML

### 💾 Export & Reporting

* **CSV Outputs:** Cleaned data, analytics results, profit summaries
* **Interactive HTML Charts:** For offline exploration
* **Validation Reports:** Automatic data-quality and integrity summaries

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

* **100 % Calculation Accuracy** — 13 / 13 validation tests passed
* **O(n) Efficiency** — Vectorized operations achieve 10–100× speedup over naive loops
* **Zero Missing Values** — Processed 5,000+ trading records with full integrity
* **Real Insights** — Identified TSLA multiple-transaction profits > 2500 %

### 📈 Analytical Insights

| Stock          | Volatility (σ) | Sharpe Ratio | Key Finding                                 |
| :------------- | :------------- | :----------- | :------------------------------------------ |
| **TSLA**       | 0.045          | 1.8          | Highest volatility, best for active trading |
| **JPM**        | 0.017          | **2.1**      | Best risk-adjusted return                   |
| **AAPL**       | 0.022          | 1.5          | Stable long-term growth                     |
| **All Stocks** | —              | —            | Avg. upward streak: 3.2 days                |

> Multiple-transaction strategies consistently outperform single transactions by 200–600 %.

---

## 🛠️ Installation

1. **Clone the Repository**

   ```bash
   git clone <repository-url>
   cd stonk-analyzer
   ```

2. **Create Virtual Environment** *(recommended)*

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
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
*Ideal for quick exploratory analysis and classroom demonstrations.*

### 💻 Option 2: Run Locally

```bash
streamlit run main.py
```

### 📓 Option 3: Notebook Exploration

```bash
jupyter notebook notebook/EDA.ipynb
```

### 🎥 Option 4: Watch Tutorial

📹 [5-Minute User Guide](https://youtu.be/kBWDgTUPgP4)

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
}
```

### Global Settings

* **📅 Date Range:** Apply globally or override per ticker
* **📊 SMA Periods:** Common presets (5, 20, 50) or custom windows
* **⚙️ Smart Defaults:** Optimized analysis ranges and logging options

---

## 🧪 Testing & Validation

Run all tests:

```bash
python src/validation.py
```

### Coverage Includes

* Daily return accuracy (pandas parity)
* SMA verification vs. manual computation
* O(n) profit optimization algorithms
* Edge-case handling and performance benchmarks
* Schema validation and data-integrity checks

**Results:** ✅ 13 / 13 tests passed — 100 % accuracy and transparency

---

## 📈 Example Workflow

1. Configure global date range and SMA windows
2. Load tickers (e.g., `AAPL, MSFT, TSLA`) via Yahoo Finance
3. Clean and validate data (0 missing values)
4. Analyze trends with SMA crossovers and buy/sell markers
5. Compute profits (single vs multiple transactions)
6. Assess risk (volatility + Sharpe ratios)
7. Visualize results interactively
8. Export CSV and HTML reports

---

## 🎓 Educational Value

This project demonstrates:

* **Modular Python Design:** Independent data, analytics, and visualization layers
* **Algorithmic Efficiency:** O(n) vectorized implementations
* **Financial Computing:** Accurate, industry-standard metrics
* **Validation-First Development:** Comprehensive testing ensures reliability
* **Accessible Web Deployment:** Streamlit for interactive, transparent dashboards

---

## 🤝 Contributing

1. Fork this repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all validations pass
5. Submit a pull request

---

## 📄 License

This project is for **educational use** under the *INF1002 Programming Fundamentals* module.
Not intended for commercial trading or financial advice.

---

## 🆘 Support

If you encounter issues:

1. 📹 Watch the [Tutorial](https://youtu.be/kBWDgTUPgP4)
2. 🧾 Check validation reports in `data/validation/`
3. ⚙️ Verify dependencies from `requirements.txt`
4. 🌐 Ensure Yahoo Finance API availability
5. 🧩 Review configuration settings in `config.py`

---

**Built with ❤️ for transparent, educational financial data analysis**
**Team INF1002 — Stonk Analyzer (2025)**
