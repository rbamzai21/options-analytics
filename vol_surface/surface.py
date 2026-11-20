import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from scipy.interpolate import griddata

from data.market_data import compute_time_to_expiry, get_options_chain, get_risk_free_rate, get_spot_price
from vol_surface.implied_vol import solve_iv

MIN_T = 1 / 365


def contracts_to_rows(chain: pd.DataFrame, s, r, t, expiry, option_type) -> list[dict]:
    rows = []
    for _, row in chain.iterrows():
        mid_price = (row["bid"] + row["ask"]) / 2
        k = row["strike"]

        iv, converged = solve_iv(mid_price, s, k, r, t, option_type)

        if converged and 0.01 < iv < 5.0:
            rows.append({
                "strike": k,
                "moneyness": k / s,
                "expiry": expiry,
                "mid_price": mid_price,
                "iv": iv,
                "t": t,
                "option_type": option_type,
            })
    return rows


def build_surface(ticker: str) -> pd.DataFrame:
    # Fetches options data across all available expiries and computes IV for each
    # liquid contract. Only out-of-the-money options are used (puts below spot,
    # calls at or above spot): they are more liquid, and using one option per
    # strike avoids conflicting call and put IVs at the same point on the surface.
    # Returns a dataframe with columns: strike, moneyness, expiry, mid_price, iv, t, option_type

    s = get_spot_price(ticker)
    r = get_risk_free_rate()
    rows = []

    for expiry in yf.Ticker(ticker).options:
        t = compute_time_to_expiry(expiry)
        if t <= MIN_T:
            continue

        calls, puts = get_options_chain(ticker, expiry)
        rows += contracts_to_rows(calls[calls["strike"] >= s], s, r, t, expiry, "call")
        rows += contracts_to_rows(puts[puts["strike"] < s], s, r, t, expiry, "put")

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
        method="cubic",
    )

    fig = go.Figure(data=[go.Surface(x=strike_grid, y=t_grid, z=iv_grid)])
    fig.update_layout(
        title="Implied Volatility Surface",
        scene=dict(
            xaxis_title="Strike",
            yaxis_title="Time to Expiry (Years)",
            zaxis_title="Implied Volatility",
        ),
    )
    return fig
