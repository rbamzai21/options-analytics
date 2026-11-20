import math

import pytest

from models.black_scholes import call_and_put_pricing
from vol_surface.implied_vol import (
    implied_vol_bisection,
    implied_vol_newton,
    no_arbitrage_bounds,
    solve_iv,
)


@pytest.mark.parametrize("k", [70, 90, 100, 110, 140])
@pytest.mark.parametrize("t", [0.05, 0.5, 2.0])
@pytest.mark.parametrize("true_sigma", [0.1, 0.3, 0.8])
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_solve_iv_recovers_true_vol(k, t, true_sigma, option_type):
    s, r = 100, 0.04
    call, put = call_and_put_pricing(s, k, r, t, true_sigma)
    price = call if option_type == "call" else put

    # Skip contracts whose time value is too small to carry information about vol
    # (deep in-the-money or deep out-of-the-money, short-dated)
    lower, _ = no_arbitrage_bounds(s, k, r, t, option_type)
    if price - lower < 1e-3:
        pytest.skip("time value too small to identify vol")

    iv, converged = solve_iv(price, s, k, r, t, option_type)
    assert converged
    assert iv == pytest.approx(true_sigma, abs=1e-4)


def test_newton_converges_quickly_near_the_money():
    price = call_and_put_pricing(100, 100, 0.05, 1.0, 0.25)[0]
    iv, converged = implied_vol_newton(price, 100, 100, 0.05, 1.0, max_iterations=5)
    assert converged
    assert iv == pytest.approx(0.25, abs=1e-6)


def test_bisection_rejects_unreachable_price():
    # A call can never be worth more than the stock
    iv, converged = implied_vol_bisection(150, 100, 100, 0.05, 1.0)
    assert not converged
    assert math.isnan(iv)


@pytest.mark.parametrize("price,valid", [(1.0, False), (54.0, False), (60.0, True), (150.0, False)])
def test_solve_iv_respects_no_arbitrage_bounds(price, valid):
    # In-the-money call: lower bound is S - K*e^(-rT) ≈ 54.88, upper bound is S = 150
    s, k, r, t = 150, 100, 0.05, 1.0
    _, converged = solve_iv(price, s, k, r, t, "call")
    assert converged == valid


def test_falls_back_to_bisection_when_newton_fails():
    # Far out-of-the-money, short-dated: vega at the initial guess is ~0
    s, k, r, t, sigma = 100, 200, 0.05, 0.1, 1.5
    price = call_and_put_pricing(s, k, r, t, sigma)[0]
    _, newton_ok = implied_vol_newton(price, s, k, r, t)
    iv, converged = solve_iv(price, s, k, r, t)
    assert not newton_ok
    assert converged
    assert iv == pytest.approx(sigma, abs=1e-4)
