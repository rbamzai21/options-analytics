from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from data.market_data import MARKET_TZ, clean_chain, compute_time_to_expiry
from models.black_scholes import call_and_put_pricing
from vol_surface.surface import contracts_to_rows, plot_surface


def test_time_to_expiry_counts_to_market_close():
    now = datetime(2026, 1, 1, 16, 0, tzinfo=MARKET_TZ)
    assert compute_time_to_expiry("2027-01-01", now=now) == pytest.approx(1.0)
    # Same-day expiry at 10am has 6 trading hours left, not zero
    morning = datetime(2026, 1, 1, 10, 0, tzinfo=MARKET_TZ)
    assert compute_time_to_expiry("2026-01-01", now=morning) == pytest.approx(6 / (365 * 24))


def test_clean_chain_filters_illiquid_contracts():
    chain = pd.DataFrame({
        "strike":       [90, 95, 100, 105, 110],
        "bid":          [10.0, 0.0, 3.0, 2.0, np.nan],
        "ask":          [10.5, 0.5, 3.2, 1.5, 1.0],
        "volume":       [50, 50, 5, 50, 50],
        "openInterest": [100, 100, 100, 100, 100],
    })
    # Keeps only strike 90: zero bid, low volume, crossed market and NaN are dropped
    assert clean_chain(chain)["strike"].tolist() == [90]


def test_contracts_to_rows_recovers_iv():
    s, r, t, sigma = 100, 0.04, 0.5, 0.3
    strikes = [100, 110, 120]
    mids = [call_and_put_pricing(s, k, r, t, sigma)[0] for k in strikes]
    chain = pd.DataFrame({"strike": strikes, "bid": mids, "ask": mids})

    rows = contracts_to_rows(chain, s, r, t, "2026-06-01", "call")
    assert len(rows) == 3
    assert all(row["iv"] == pytest.approx(sigma, abs=1e-4) for row in rows)


def test_plot_surface_builds_figure():
    strikes, times = np.meshgrid(np.linspace(80, 120, 9), np.linspace(0.1, 1.0, 5))
    df = pd.DataFrame({
        "strike": strikes.ravel(),
        "t": times.ravel(),
        "iv": 0.2 + 0.001 * (strikes.ravel() - 100) ** 2 / 10,
    })
    fig = plot_surface(df)
    assert fig.data[0].type == "surface"
