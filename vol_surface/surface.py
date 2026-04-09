import pandas as pd
import sys
import os
import plotly.graph_objects as go
from scipy.interpolate import griddata
import numpy as np
import yfinance as yf
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.market_data import get_spot_price, get_risk_free_rate, get_options_chain, compute_time_to_expiry
from vol_surface.implied_vol import implied_vol_newton

def build_surface(ticker: str) -> pd.DataFrame:
    # Fetches options data across all available expiries
    # Computes IV for each contract using implied_vol solver
    # returns dataframe with columns: strike, expiry, mid_price, iv, t, option_type
    
    ticker_obj = yf.Ticker(ticker)
    expiries = ticker_obj.options

    try:
        s = get_spot_price(ticker)
    except ValueError as e:
        raise ValueError(str(e))

    r = get_risk_free_rate()
    rows = []

    for e in expiries:
        calls, puts = get_options_chain(ticker, e)
        t = compute_time_to_expiry(e)

        if t <= 0: continue

        for _, row in calls.iterrows():
            mid_price = (row["bid"] + row["ask"]) / 2
            k = row["strike"]

            iv, convergence = implied_vol_newton(mid_price, s, k, r, t, "call")

            if convergence and 0.01 < iv < 5.0:
                rows.append({
                    "strike": k, 
                    "expiry": e, 
                    "mid_price": mid_price, 
                    "iv": iv, 
                    "t": t, 
                    "option_type": "call"
                })

        for _, row in puts.iterrows():
            mid_price = (row["bid"] + row["ask"]) / 2
            k = row["strike"]

            iv, convergence = implied_vol_newton(mid_price, s, k, r, t, "put")

            if convergence and 0.01 < iv < 5.0:
                rows.append({
                    "strike": k, 
                    "expiry": e, 
                    "mid_price": mid_price, 
                    "iv": iv, 
                    "t": t, 
                    "option_type": "put"
                })
    return pd.DataFrame(rows)

def plot_surface(df: pd.DataFrame):
    # Plots the vol surface as a 3D surface plot
    # x-axis: strike, y-axis: time to expiry, z-axis: implied vol
    
    strike_grid = np.linspace(df["strike"].min(), df["strike"].max(), 50)
    t_grid = np.linspace(df["t"].min(), df["t"].max(), 50)
    strike_mesh, t_mesh = np.meshgrid(strike_grid, t_grid)

    iv_grid = griddata(
        points=(df["strike"], df["t"]), 
        values=df["iv"], 
        xi=(strike_mesh, t_mesh),
        method="cubic"
    )

    fig = go.Figure(data=[go.Surface(x=strike_grid, y=t_grid, z=iv_grid)])
    fig.update_layout(
        title="Implied Volatility Surface", 
        scene=dict(
            xaxis_title="Strike",
            yaxis_title="Time to Expiry (Years)", 
            zaxis_title="Implied Volatility"
        )
    )
    return fig

