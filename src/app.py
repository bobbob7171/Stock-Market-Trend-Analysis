"""
Streamlit App Interface
Provides interactive UI for selecting stocks and parameters.
"""

import streamlit as st
import pandas as pd


def main():
    st.title("📈 Stock Market Trend Analysis")
    st.write("Explore SMA, streaks, and profit opportunities.")

    # Example UI placeholders
    ticker = st.text_input("Enter Stock Ticker", "AAPL")
    sma_window = st.slider("SMA Window", 5, 50, 20)

    st.write("Charts and results will appear here...")

if __name__ == "__main__":
    main()
