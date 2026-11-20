import os
import sys

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# `streamlit run dashboard/app.py` puts dashboard/ on the path, not the repo root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.market_data import compute_time_to_expiry, get_options_chain, get_risk_free_rate, get_spot_price
from greeks.analytical import compute_all
from models import binomial_tree, monte_carlo
from models.black_scholes import call_and_put_pricing
from vol_surface.implied_vol import no_arbitrage_bounds, solve_iv
from vol_surface.surface import build_surface, plot_surface

# Market data is cached so changing a widget doesn't re-download everything
# (Streamlit reruns the whole script on every interaction).
CACHE_TTL = 300  # seconds


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def cached_spot(ticker):
    return get_spot_price(ticker)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def cached_rate():
    return get_risk_free_rate()


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def cached_expiries(ticker):
    return list(yf.Ticker(ticker).options)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def cached_chain(ticker, expiry):
    return get_options_chain(ticker, expiry)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def cached_surface(ticker):
    return build_surface(ticker)


st.set_page_config(page_title="Options Analytics", layout="wide")
page = st.sidebar.selectbox("Navigate", ["Pricer & Greeks", "Vol Surface"])

if page == "Pricer & Greeks":
    st.title("Option Pricer & Greeks")

    st.sidebar.header("Parameters")
    ticker = st.sidebar.text_input("Ticker", value="AAPL").strip().upper()

    if ticker:
        try:
            S = cached_spot(ticker)
            r = cached_rate()
            st.sidebar.metric("Spot Price", f"${S:.2f}")
            st.sidebar.metric("Risk-Free Rate", f"{r*100:.2f}%")
        except Exception as e:
            st.error(f"Could not fetch data for {ticker}: {e}")
            st.stop()

        expiries = [e for e in cached_expiries(ticker) if compute_time_to_expiry(e) > 1 / 365]
        if len(expiries) == 0:
            st.error("No options data available for this ticker.")
            st.stop()

        expiry = st.sidebar.selectbox("Expiry Date", expiries)
        T = compute_time_to_expiry(expiry)

        opt_type = st.sidebar.selectbox("Option Type", ["Call", "Put"]).lower()
        model = st.sidebar.selectbox("Model", ["Black-Scholes", "Binomial Tree", "Monte Carlo"])

        calls, puts = cached_chain(ticker, expiry)
        chain = calls if opt_type == "call" else puts

        if len(chain) == 0:
            st.warning("No liquid contracts found for this expiry")
            st.stop()

        # Default to the strike closest to the current spot price (at the money)
        strikes = chain["strike"].tolist()
        atm_index = int(np.argmin([abs(k - S) for k in strikes]))
        K = st.sidebar.selectbox("Strike", strikes, index=atm_index)

        row = chain[chain["strike"] == K].iloc[0]
        bid = row["bid"]
        ask = row["ask"]
        mid = (bid + ask) / 2
        volume = row["volume"]
        open_interest = row["openInterest"]

        iv, converged = solve_iv(mid, S, K, r, T, opt_type)
        if not converged:
            lower, upper = no_arbitrage_bounds(S, K, r, T, opt_type)
            st.warning(
                f"Could not compute implied volatility for this contract. The mid price "
                f"${mid:.2f} may be outside no-arbitrage bounds (${lower:.2f} – ${upper:.2f})."
            )
            st.stop()
        sigma = iv

        std_err = None
        if model == "Black-Scholes":
            call_price, put_price = call_and_put_pricing(S, K, r, T, sigma)
            model_price = call_price if opt_type == "call" else put_price
        elif model == "Binomial Tree":
            # Listed US equity options are American-style
            model_price = binomial_tree.price(S, K, r, T, sigma, n=500, option_type=opt_type, exercise="american")
        else:
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
        col3.metric("Implied Vol", f"{sigma*100:.2f}%")

        if model == "Binomial Tree":
            st.caption("Binomial tree prices American exercise (500 steps). Black-Scholes and Monte Carlo are European.")
        if std_err is not None:
            st.caption(f"Std Error: ±{std_err:.4f}")

        st.subheader("Greeks")
        greeks = compute_all(S, K, r, T, sigma)
        g = greeks[opt_type]

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Delta", f"{g['delta']:.4f}")
        col2.metric("Gamma", f"{g['gamma']:.4f}")
        col3.metric("Vega (per 1% vol)", f"{g['vega']:.4f}")
        col4.metric("Theta (per day)", f"{g['theta']:.4f}")
        col5.metric("Rho (per 1.0 rate)", f"{g['rho']:.4f}")

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
        st.plotly_chart(fig, width="stretch")

elif page == "Vol Surface":
    st.title("Implied Volatility Surface")

    ticker = st.text_input("Ticker", value="AAPL").strip().upper()

    if st.button("Build Surface"):
        try:
            with st.spinner("Fetching options data... this may take a while"):
                df = cached_surface(ticker)
            if len(df) < 10:
                st.error("Not enough liquid contracts to build a surface for this ticker.")
                st.stop()
            st.success(f"Got {len(df)} out-of-the-money contracts")
            fig = plot_surface(df)
            st.plotly_chart(fig, width="stretch")
            st.subheader("Raw Data")
            st.dataframe(df)
        except ValueError as e:
            st.error(str(e))
