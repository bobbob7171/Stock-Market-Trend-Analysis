# graph.py

import pandas as pd
import plotly.graph_objects as go
from typing import List, Tuple
import streamlit as st


# # ---------------------------
# # Helper: Show Figure
# # ---------------------------
# def show_fig(fig: go.Figure, title: str = "") -> None:
#     """Helper function to display a Plotly figure in the browser."""
#     if title:
#         fig.update_layout(title=title)
#     fig.show(renderer="browser")

def show_fig(fig: go.Figure, title: str = "") -> None:
    """Display a Plotly figure directly in Streamlit."""
    if title:
        fig.update_layout(title=title)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# 1. Price + SMA
# ---------------------------
def plot_price_sma(df: pd.DataFrame, ticker: str, sma_windows: List[int] = [20, 50]) -> None:
    """
    Plot closing prices and SMA overlays for a given ticker.

    Args:
        df (pd.DataFrame): Must contain ['Date', 'Ticker', 'Close'].
        ticker (str): Ticker symbol to plot.
        sma_windows (list[int], optional): List of SMA window sizes.
    """
    subset = df[df["Ticker"] == ticker].sort_values("Date")
    
    fig = go.Figure()
    # Closing price
    fig.add_trace(go.Scatter(
        x=subset["Date"], y=subset["Close"], mode="lines",
        name=f"{ticker} Close", line=dict(color="blue"),
        hovertemplate="Date: %{x}<br>Close: %{y:$,.2f}<extra></extra>"
    ))
    
    # SMAs
    for window in sma_windows:
        sma_col = f"SMA_{window}"
        if sma_col in subset.columns:
            fig.add_trace(go.Scatter(
                x=subset["Date"], y=subset[sma_col], mode="lines",
                name=f"SMA {window}", line=dict(dash="dash"),
                hovertemplate="Date: %{x}<br>SMA: %{y:$,.2f}<extra></extra>"
            ))
    
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Price",
        template="plotly_white", hovermode="x unified"
    )
    show_fig(fig, title=f"{ticker} Closing Prices & SMAs")


# ---------------------------
# 2. Highlight Streaks
# ---------------------------
def plot_streaks(df: pd.DataFrame, ticker: str) -> None:
    """
    Highlight positive/negative streaks on a stock chart.

    Args:
        df (pd.DataFrame): Must contain ['Date', 'Ticker', 'Close', 'Streak'].
        ticker (str): Ticker symbol to plot.
    """
    subset = df[df["Ticker"] == ticker].sort_values("Date")
    
    fig = go.Figure()
    # Close line
    fig.add_trace(go.Scatter(
        x=subset["Date"], y=subset["Close"], mode="lines",
        name=f"{ticker} Close", line=dict(color="blue")
    ))
    
    # Positive & Negative streaks
    for streak_type, color in [("Positive", "green"), ("Negative", "red")]:
        mask = subset["Streak"].apply(lambda x: x > 0 if streak_type=="Positive" else x < 0)
        if mask.any():
            fig.add_trace(go.Scatter(
                x=subset.loc[mask, "Date"], y=subset.loc[mask, "Close"],
                mode="markers", marker=dict(color=color, size=7),
                name=f"{streak_type} Streak",
                hovertemplate="Date: %{x}<br>Close: %{y:$,.2f}<extra></extra>"
            ))
    
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Price",
        template="plotly_white", hovermode="x unified"
    )
    show_fig(fig, title=f"{ticker} Closing Prices with Streaks")


# ---------------------------
# 3. Annotate Buy/Sell Trades
# ---------------------------
def annotate_profits(df: pd.DataFrame, ticker: str, trades: List[Tuple[int, int]]) -> None:
    """
    Annotate buy/sell points for profit optimization.

    Args:
        df (pd.DataFrame): Must contain ['Date', 'Ticker', 'Close'].
        ticker (str): Ticker symbol to plot.
        trades (list[tuple[int,int]]): List of (buy_index, sell_index) tuples.
    """
    subset = df[df["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=subset["Date"], y=subset["Close"], mode="lines", name=f"{ticker} Close"
    ))
    
    for buy_idx, sell_idx in trades:
        fig.add_trace(go.Scatter(
            x=[subset.iloc[buy_idx]["Date"]], y=[subset.iloc[buy_idx]["Close"]],
            mode="markers+text", marker=dict(color="green", size=12, symbol="triangle-up"),
            text=["Buy"], textposition="top center", name="Buy"
        ))
        fig.add_trace(go.Scatter(
            x=[subset.iloc[sell_idx]["Date"]], y=[subset.iloc[sell_idx]["Close"]],
            mode="markers+text", marker=dict(color="red", size=12, symbol="triangle-down"),
            text=["Sell"], textposition="bottom center", name="Sell"
        ))
    
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Price",
        template="plotly_white", hovermode="x unified"
    )
    show_fig(fig, title=f"{ticker} Buy/Sell Trades")


# ---------------------------
# 4. Profit Comparison
# ---------------------------
def plot_profit_comparison(df: pd.DataFrame) -> None:
    """
    Interactive bar chart comparing single vs multiple transaction profits per ticker.

    Args:
        df (pd.DataFrame): Must contain ['Ticker', 'MaxProfit_Single', 'MaxProfit_Multiple'].
    """
    if df.empty:
        print("Profit DataFrame is empty. Skipping plot.")
        return

    required_cols = {"Ticker", "MaxProfit_Single", "MaxProfit_Multiple"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

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
            hovertemplate="%{x}<br>"+name+": %{y:.2f}<extra></extra>"
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
