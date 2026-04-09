import streamlit as st
import numpy as np
import plotly.graph_objects as go
import datetime
import sys 
import os
import yfinance as yf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.black_scholes import call_and_put_pricing
from models import binomial_tree, monte_carlo
from greeks.analytical import compute_all
from vol_surface.surface import build_surface, plot_surface
from data.market_data import compute_time_to_expiry, get_spot_price, get_risk_free_rate, get_options_chain
from vol_surface.implied_vol import implied_vol_newton, implied_vol_bisection

st.set_page_config(page_title="Options Analytics", layout="wide")
page = st.sidebar.selectbox("Navigate", ["Pricer & Greeks", "Vol Surface"])

if page == "Pricer & Greeks":
    st.title("Option Pricer & Greeks")

    st.sidebar.header("Parameters")
    ticker = st.sidebar.text_input("Ticker", value="AAPL")

    if ticker:
        ticker_obj = yf.Ticker(ticker)

        try:
            S = get_spot_price(ticker)
            r = get_risk_free_rate()
            st.sidebar.metric("Spot Price", f"${S:.2f}")
            st.sidebar.metric("Risk-Free Rate", f"{r*100:.2f}%")
        except Exception as e:
            st.error(f"Could not fetch data for {ticker}: {e}")
            st.stop()

        expiries = ticker_obj.options
        expiries = [e for e in ticker_obj.options if compute_time_to_expiry(e) > 1/365]
        if len(expiries) == 0:
            st.error("No options data available for this ticker.")
            st.stop()

        expiry = st.sidebar.selectbox("Expiry Date", expiries)
        T = compute_time_to_expiry(expiry)

        opt_type = st.sidebar.selectbox("Option Type", ["Call", "Put"])
        model = st.sidebar.selectbox("Model", ["Black-Scholes", "Binomial Tree", "Monte Carlo"])

        calls, puts = get_options_chain(ticker, expiry)
        opt_type = opt_type.lower()
        chain = calls if opt_type == "call" else puts

        if len(chain) == 0:
            st.warning("No liquid contracts found for this expiry")
            st.stop()
        
        strikes = chain["strike"].tolist()
        K = st.sidebar.selectbox("Strike", strikes)

        row = chain[chain["strike"] == K].iloc[0]
        bid = row["bid"]
        ask = row["ask"]
        mid = (bid + ask) / 2
        volume = row["volume"]
        open_interest = row["openInterest"]

        if T < 1/365:
            st.warning("Contract expires too soon to compute IV reliably.")
            st.stop()
        intrinsic = max(S - K, 0) if opt_type == "call" else max(K - S, 0)
        
        if mid <= 0:
            st.warning("Invalid mid price — no market for this contract.")
            st.stop()

        if mid <= intrinsic * 0.5:
            st.warning("Mid price below intrinsic value — contract may be mispriced.")
            st.stop()

        iv, converged = implied_vol_newton(mid, S, K, r, T, opt_type)
        
        if not converged:
            iv, converged = implied_vol_bisection(mid, S, K, r, T, opt_type)

        sigma = iv if converged else None   
        if sigma is None:
            st.warning("Could not compute implied volatility for this contract.")
            st.stop()

        if model == "Black-Scholes":
            call_price, put_price = call_and_put_pricing(S, K, r, T, sigma)
            model_price = call_price if opt_type == "call" else put_price

        elif model == "Binomial Tree":
            model_price = binomial_tree.price(S, K, r, T, sigma, n=100, option_type=opt_type)
        elif model == "Monte Carlo":
            model_price, std_err = monte_carlo.price(S, K, r, T, sigma, option_type=opt_type)
        
        st.subheader("Market Data")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Bid", f"${bid:.2f}")
        col2.metric("Ask", f"${ask:.2f}")
        col3.metric("Mid", f"${mid:.2f}")
        col4.metric("Volume", f"{int(volume):,}")
        col5.metric("Open Interest", f"{int(open_interest):,}")

        st.subheader("Model Output")
        col1, col2, col3 = st.columns(3)
        col1.metric("Model Price", f"${model_price:.4f}")
        col2.metric("Market Mid", f"${mid:.4f}")
        col3.metric("Implied Vol", f"{sigma*100:.2f}%" if converged else "Did not converge")

        if model == "Monte Carlo":
            st.caption(f"Std Error: ±{std_err:.4f}")
        
        st.subheader("Greeks")
        greeks = compute_all(S, K, r, T, sigma)
        g = greeks[opt_type]

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Delta", f"{g['delta']:.4f}")
        col2.metric("Gamma", f"{g['gamma']:.4f}")
        col3.metric("Vega (per 1% vol)", f"{g['vega']:.4f}")
        col4.metric("Theta (per day)", f"{g['theta']:.4f}")
        col5.metric("Rho", f"{g['rho']:.4f}")

        st.subheader("Payoff at Expiry")
        spots = np.linspace(S * 0.5, S * 1.5, 200)
        if opt_type == "call":
            payoffs = np.maximum(spots - K, 0)
        else:
            payoffs = np.maximum(K - spots, 0)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=spots, y=payoffs, mode="lines", name="Payoff"))
        fig.add_vline(x=S, line_dash="dash", line_color="gray", annotation_text="Current Spot")
        fig.add_vline(x=K, line_dash="dash", line_color="red", annotation_text="Strike")
        fig.update_layout(xaxis_title="Stock Price at Expiry", yaxis_title="Payoff ($)")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Vol Surface":
    st.title("Implied Volatility Surface")

    ticker = st.text_input("Ticker", value="AAPL")

    if st.button("Build Surface"):
        try:
            with st.spinner("Fetching options data... this may take a while"):
                df = build_surface(ticker)
            st.success(f"Got {len(df)} contracts")
            fig = plot_surface(df)
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Raw Data")
            st.dataframe(df)
        except ValueError as e:
            st.error(str(e))
