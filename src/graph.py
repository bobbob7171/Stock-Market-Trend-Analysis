"""
graph.py

Module for generating Plotly visualizations for stock analysis:
- Price + SMA with buy/sell markers
- Upward/Downward trend highlighting
- Profit comparison
- Annualized risk-return scatter plot
- Best buy/sell points for multiple transactions
"""

import pandas as pd
import plotly.graph_objects as go
from typing import List, Optional
import streamlit as st
import plotly.express as px


# Helper: Hover Template
def hover_template(y_label: str = "Value", currency: bool = False, decimals: int = 2) -> str:
    """Generate a Plotly hover template."""
    if currency:
        return f"Date: %{{x}}<br>{y_label}: $%{{y:,.{decimals}f}}<extra></extra>"
    else:
        return f"Date: %{{x}}<br>{y_label}: %{{y:.{decimals}f}}<extra></extra>"


# Helper: Show Figure in Streamlit
def show_fig(fig: go.Figure, title: str = "", height: int = 600) -> None:
    """Display a Plotly figure in Streamlit."""
    if title:
        fig.update_layout(title=title)
    fig.update_layout(height=height)
    st.plotly_chart(fig, use_container_width=True)

# Price + SMA + Buy/Sell Crossovers
def plot_price_sma(df: pd.DataFrame, ticker: str, sma_windows: List[int] = [20, 50], show_markers: bool = True) -> Optional[go.Figure]:
    """
    Plot closing price with multiple SMA overlays and buy/sell crossover markers.

    Args:
        df (pd.DataFrame): Must contain ['Date', 'Ticker', 'Close'].
        ticker (str): Ticker symbol to plot.
        sma_windows (List[int]): List of SMA window sizes to plot.
        show_markers (bool): Whether to display point markers on the closing price line.
    """
    subset = df[df["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)
    if subset.empty:
        st.warning(f"No data found for ticker {ticker}")
        return None

    fig = go.Figure()


    # 1. Plot Closing Price
    fig.add_trace(go.Scatter(
        x=subset["Date"], y=subset["Close"], mode="lines",
        name=f"{ticker} Close", line=dict(color="blue"),
        hovertemplate=hover_template("Close", currency=True)
    ))


    # 2. Plot SMA lines
    dash_styles = ["dash", "dot", "dashdot", "longdash", "longdashdot"]
    for i, window in enumerate(sma_windows):
        sma_col = f"SMA_{window}"
        if sma_col not in subset.columns:
            subset[sma_col] = subset["Close"].rolling(window).mean()

        # Add SMA line
        fig.add_trace(go.Scatter(
            x=subset["Date"], y=subset[sma_col],
            mode="lines", name=f"SMA {window}",
            line=dict(dash=dash_styles[i % len(dash_styles)]),
            hovertemplate=hover_template(f"SMA {window}", currency=True)
        ))

    
        # 3. Detect Crossings (Buy/Sell)
        subset[f"Crossings_{window}"] = 0
        subset.loc[
            (subset["Close"] > subset[sma_col]) &
            (subset["Close"].shift(1) <= subset[sma_col].shift(1)),
            f"Crossings_{window}"
        ] = 1  # Buy signal
        subset.loc[
            (subset["Close"] < subset[sma_col]) &
            (subset["Close"].shift(1) >= subset[sma_col].shift(1)),
            f"Crossings_{window}"
        ] = -1  # Sell signal

        # Buy markers (green)
        cross_above = subset[subset[f"Crossings_{window}"] == 1]
        if not cross_above.empty:
            fig.add_trace(go.Scatter(
                x=cross_above["Date"], y=cross_above[sma_col],
                mode="markers", name=f"Buy Signal (SMA {window})",
                marker=dict(color="green", size=8, symbol="triangle-up")
            ))

        # Sell markers (red)
        cross_below = subset[subset[f"Crossings_{window}"] == -1]
        if not cross_below.empty:
            fig.add_trace(go.Scatter(
                x=cross_below["Date"], y=cross_below[sma_col],
                mode="markers", name=f"Sell Signal (SMA {window})",
                marker=dict(color="red", size=8, symbol="triangle-down")
            ))


    # 4. Layout
    fig.update_layout(
        title=f"{ticker} Closing Prices with SMA Crossovers",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        hovermode="x unified"
    )

    show_fig(fig, title=f"{ticker} Price + SMA Crossovers")
    return fig


# 2. Highlight Upward/Downward Trends 
def plot_streaks(df: pd.DataFrame, ticker: str) -> Optional[go.Figure]:
    subset = df[df["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)
    if subset.empty:
        st.warning(f"No data found for ticker {ticker}")
        return None

    fig = go.Figure()

    # Plot Close Price Line
    fig.add_trace(go.Scatter(
        x=subset["Date"],
        y=subset["Close"],
        mode="lines",
        name=f"{ticker} Close",
        line=dict(color="blue"),
        hovertemplate=hover_template("Close", currency=True)
    ))

    # Prepare upward and downward trend segments
    up_x, up_y = [], []
    down_x, down_y = [], []

    for i in range(1, len(subset)):
        if subset["Close"][i] > subset["Close"][i-1]:
            # Upward trend
            up_x.extend([subset["Date"][i-1], subset["Date"][i], None])
            up_y.extend([subset["Close"][i-1], subset["Close"][i], None])
        elif subset["Close"][i] < subset["Close"][i-1]:
            # Downward trend
            down_x.extend([subset["Date"][i-1], subset["Date"][i], None])
            down_y.extend([subset["Close"][i-1], subset["Close"][i], None])

    # Add upward trend line (all segments as one trace)
    if up_x:
        fig.add_trace(go.Scatter(
            x=up_x,
            y=up_y,
            mode="lines",
            line=dict(color="green", width=4),
            name="Upward Trend",
            hoverinfo="skip"
        ))

    # Add downward trend line (all segments as one trace)
    if down_x:
        fig.add_trace(go.Scatter(
            x=down_x,
            y=down_y,
            mode="lines",
            line=dict(color="red", width=4),
            name="Downward Trend",
            hoverinfo="skip"
        ))

    # Layout
    fig.update_layout(
        title=f"{ticker} Closing Prices with Upward/Downward Trends",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        hovermode="x unified"
    )

    show_fig(fig, title=f"{ticker} Closing Prices with Upward/Downward Trends")
    return fig


# 3. Profit Comparison
def plot_profit_comparison(df: pd.DataFrame) -> Optional[go.Figure]:
    if df.empty:
        st.warning("Profit DataFrame is empty. Skipping plot.")
        return None

    required_cols = {"Ticker", "MaxProfit_Single", "MaxProfit_Multiple"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    df = df.fillna(0).sort_values("MaxProfit_Multiple", ascending=False)

    fig = go.Figure()
    for col, name, color in zip(
        ["MaxProfit_Single", "MaxProfit_Multiple"],
        ["Single Transaction", "Multiple Transactions"],
        ["royalblue", "orange"]
    ):
        fig.add_trace(go.Bar(
            x=df["Ticker"],
            y=df[col],
            name=name,
            marker_color=color,
            text=[f"{val:.2f}" for val in df[col]],
            textposition="auto",
            hovertemplate=f"%{{x}}<br>{name}: %{{y:.2f}}<extra></extra>"
        ))

    fig.update_layout(
        xaxis_title="Ticker",
        yaxis_title="Profit",
        barmode="group",
        template="plotly_white",
        hovermode="x unified",
        yaxis=dict(autorange=True, title_standoff=10)
    )
    show_fig(fig, title="Profit Comparison: Single vs Multiple Transactions")
    return fig


# 4. Annual Risk-Return Plot
def plot_annual_risk_return(risk_return_df: pd.DataFrame) -> Optional[go.Figure]:
    """
    Plot annualized risk vs return for each ticker, highlighting the best Sharpe ratio.
    
    Args:
        risk_return_df (pd.DataFrame): Must contain ['Ticker', 'Annual_Return', 'Annual_Volatility', 'Sharpe_Ratio']
    """
    if risk_return_df.empty:
        st.warning("Risk-return DataFrame is empty. Cannot plot.")
        return None

    # Assign a unique color to each ticker
    tickers_unique = risk_return_df["Ticker"].unique()
    colors = px.colors.qualitative.Alphabet  # 26 colors
    color_map = {ticker: colors[i % len(colors)] for i, ticker in enumerate(tickers_unique)}

    best_idx = risk_return_df["Sharpe_Ratio"].idxmax()
    best_stock = risk_return_df.loc[best_idx]

    fig = go.Figure()

    # Plot all tickers individually
    for ticker in tickers_unique:
        subset = risk_return_df[risk_return_df["Ticker"] == ticker]
        fig.add_trace(go.Scatter(
            x=subset["Annual_Volatility"],
            y=subset["Annual_Return"],
            mode="markers+text",
            text=subset["Ticker"],
            textposition="top center",
            marker=dict(size=12, color=color_map[ticker]),
            name=ticker,
            hovertemplate=(
                "Ticker: %{text}<br>"
                "Return: %{y:.2%}<br>"
                "Volatility: %{x:.2%}<br>"
                "Sharpe: %{customdata:.2f}<extra></extra>"
            ),
            customdata=subset["Sharpe_Ratio"]
        ))

    # Highlight best Sharpe stock (star marker)
    fig.add_trace(go.Scatter(
        x=[best_stock["Annual_Volatility"]],
        y=[best_stock["Annual_Return"]],
        mode="markers",
        marker=dict(size=16, color="red", symbol="star"),
        name="Best Sharpe",
        hovertemplate=(
            "Best Sharpe!<br>"
            "Ticker: %{customdata[0]}<br>"
            "Return: %{y:.2%}<br>"
            "Volatility: %{x:.2%}<br>"
            "Sharpe: %{customdata[1]:.2f}<extra></extra>"
        ),
        customdata=[[best_stock["Ticker"], best_stock["Sharpe_Ratio"]]]
    ))

    fig.update_layout(
        title="Annualized Risk vs Return (Sharpe Highlighted)",
        xaxis_title="Annualized Volatility (Risk)",
        yaxis_title="Annualized Return",
        template="plotly_white",
        hovermode="closest"
    )

    show_fig(fig)
    return fig


# Best Buy/Sell Points (Multiple Transactions)
def plot_best_buy_sell(df, ticker: str):
    subset = df[df["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)
    prices = subset["Close"].values
    dates = subset["Date"]

    buys = []
    sells = []

    i = 0
    n = len(prices)
    while i < n - 1:
        # Find local minima (buy point)
        while i < n - 1 and prices[i + 1] <= prices[i]:
            i += 1
        if i == n - 1:
            break
        buy_date, buy_price = dates[i], prices[i]

        # Find local maxima (sell point)
        while i < n - 1 and prices[i + 1] >= prices[i]:
            i += 1
        sell_date, sell_price = dates[i], prices[i]

        buys.append((buy_date, buy_price))
        sells.append((sell_date, sell_price))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=prices, mode="lines", name="Close", line=dict(color="blue")))

    if buys:
        fig.add_trace(go.Scatter(
            x=[b[0] for b in buys], y=[b[1] for b in buys],
            mode="markers", name="Buy", marker=dict(color="green", symbol="triangle-up", size=8)
        ))
    if sells:
        fig.add_trace(go.Scatter(
            x=[s[0] for s in sells], y=[s[1] for s in sells],
            mode="markers", name="Sell", marker=dict(color="red", symbol="triangle-down", size=8)
        ))

    fig.update_layout(
        title=f"{ticker} — Best Buy/Sell Points (Multiple Transactions)",
        xaxis_title="Date", yaxis_title="Price",
        template="plotly_white", hovermode="x unified"
    )

    show_fig(fig)
    return fig
