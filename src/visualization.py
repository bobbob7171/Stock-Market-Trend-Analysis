"""
Visualization Module
Handles plotting of prices, SMAs, streaks, and profits.
"""

import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def plot_price_sma_interactive(dates, prices, sma, save_folder="Graphs", filename="price_sma.html"):
    # Create folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    fig = go.Figure()

    # Closing price
    fig.add_trace(go.Scatter(
        x=dates,
        y=prices,
        mode='lines',
        name='Close Price',
        line=dict(color='green', width=2)
    ))

    # SMA line
    fig.add_trace(go.Scatter(
        x=dates,
        y=sma,
        mode='lines',
        name=sma.name,
        line=dict(color='purple', width=2, dash='dash')
    ))

    # Layout
    fig.update_layout(
        title='Stock Prices with SMA Overlay',
        xaxis_title='Date',
        yaxis_title='Closing Price',
        template='plotly_white',
        width=1200,
        height=600
    )

    # Show interactive chart
    fig.show()

    # Save interactive HTML
    save_path = os.path.join(save_folder, filename)
    fig.write_html(save_path)
    print(f"Interactive chart saved to: {save_path}")
    pass



def highlight_streaks_interactive(df, dates, prices, save_folder="Graphs", filename="highlighted_streaks.html"):
    """Highlight upward/downward price movements interactively and export as HTML."""
    
    # Ensure folder exists
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    df['Change'] = df[prices].diff().fillna(0)

    fig = go.Figure()

    # Loop through each segment (day-to-day)
    for i in range(1, len(df)):
        color = 'green' if df['Change'].iloc[i] > 0 else 'red'
        fig.add_trace(go.Scatter(
            x=df[dates].iloc[i-1:i+1],
            y=df[prices].iloc[i-1:i+1],
            mode='lines',
            line=dict(color=color, width=3),
            showlegend=False
        ))

    fig.update_layout(
        title="Close Price - Green for Up, Red for Down",
        xaxis_title="Date",
        yaxis_title="Close Price",
        template="plotly_white",
        width=1200,
        height=600
    )

    # Show interactive chart
    fig.show()

    # Save interactive HTML
    save_path = os.path.join(save_folder, filename)
    fig.write_html(save_path)
    print(f"Interactive chart saved to: {save_path}")
    pass


def annotate_profits_interactive(df, dates, prices, sma, save_folder="Graphs", filename="buy_sell_signals.html"):
    """Annotate buy/sell points for profit optimization and export interactive chart."""

    # Ensure folder exists
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    buy_signals = []
    sell_signals = []
    flag = -1  # -1: no position, 1: holding, 0: sold

    for i in range(len(df)):
        # Buy signal: Close crosses above SMA
        if df[prices].iloc[i] > df[sma].iloc[i]:
            if flag != 1:  # only buy if not already holding
                buy_signals.append(df[prices].iloc[i])
                sell_signals.append(np.nan)
                flag = 1
            else:
                buy_signals.append(np.nan)
                sell_signals.append(np.nan)

        # Sell signal: Close crosses below SMA
        elif df[prices].iloc[i] < df[sma].iloc[i]:
            if flag != 0:  # only sell if holding
                buy_signals.append(np.nan)
                sell_signals.append(df[prices].iloc[i])
                flag = 0
            else:
                buy_signals.append(np.nan)
                sell_signals.append(np.nan)
        else:
            buy_signals.append(np.nan)
            sell_signals.append(np.nan)

    df['Buy'] = buy_signals
    df['Sell'] = sell_signals

    fig = go.Figure()

    # Plot Close Price
    fig.add_trace(go.Scatter(
        x=df[dates],
        y=df[prices],
        mode='lines',
        name='Close Price',
        line=dict(color='blue', width=2)
    ))

    # Plot SMA line
    fig.add_trace(go.Scatter(
        x=df[dates],
        y=df[sma],
        mode='lines',
        name=sma,
        line=dict(color='orange', width=2, dash='dash')
    ))

    # Plot Buy signals
    fig.add_trace(go.Scatter(
        x=df[dates],
        y=df['Buy'],
        mode='markers',
        name='Buy Signal',
        marker=dict(symbol='triangle-up', color='green', size=12)
    ))

    # Plot Sell signals
    fig.add_trace(go.Scatter(
        x=df[dates],
        y=df['Sell'],
        mode='markers',
        name='Sell Signal',
        marker=dict(symbol='triangle-down', color='red', size=12)
    ))

    # Layout
    fig.update_layout(
        title="Stock Buy/Sell Signals Based on SMA Strategy",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        width=1300,
        height=600
    )

    # Show interactive chart
    fig.show()

    # Save interactive HTML
    save_path = os.path.join(save_folder, filename)
    fig.write_html(save_path)
    print(f"Interactive chart saved to: {save_path}")
    pass

