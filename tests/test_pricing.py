import math

import pytest

from models import binomial_tree, monte_carlo
from models.black_scholes import call_and_put_pricing

S, K, R, T, SIGMA = 100.0, 100.0, 0.05, 1.0, 0.2
BS_CALL, BS_PUT = 10.4506, 5.5735


def test_black_scholes_reference_values():
    call, put = call_and_put_pricing(S, K, R, T, SIGMA)
    assert call == pytest.approx(BS_CALL, abs=1e-4)
    assert put == pytest.approx(BS_PUT, abs=1e-4)


@pytest.mark.parametrize("s,k,r,t,sigma", [
    (100, 100, 0.05, 1.0, 0.2),
    (100, 80, 0.03, 0.25, 0.4),
    (50, 70, 0.01, 2.0, 0.15),
])
def test_put_call_parity(s, k, r, t, sigma):
    call, put = call_and_put_pricing(s, k, r, t, sigma)
    assert (call - put) == pytest.approx(s - k * math.exp(-r * t), abs=1e-10)


def test_black_scholes_edge_cases():
    assert call_and_put_pricing(110, 100, 0.05, 0.0, 0.2) == (10, 0)
    call, put = call_and_put_pricing(110, 100, 0.05, 1.0, 0.0)
    assert call == pytest.approx(110 - 100 * math.exp(-0.05))
    assert put == 0


@pytest.mark.parametrize("option_type,expected", [("call", BS_CALL), ("put", BS_PUT)])
def test_binomial_converges_to_black_scholes(option_type, expected):
    price = binomial_tree.price(S, K, R, T, SIGMA, n=1000, option_type=option_type)
    assert price == pytest.approx(expected, abs=0.005)


def test_american_put_worth_more_than_european():
    european = binomial_tree.price(S, K, R, T, SIGMA, n=500, option_type="put", exercise="european")
    american = binomial_tree.price(S, K, R, T, SIGMA, n=500, option_type="put", exercise="american")
    assert american > european


def test_american_call_equals_european_without_dividends():
    european = binomial_tree.price(S, K, R, T, SIGMA, n=500, option_type="call", exercise="european")
    american = binomial_tree.price(S, K, R, T, SIGMA, n=500, option_type="call", exercise="american")
    assert american == pytest.approx(european, abs=1e-9)


def test_binomial_rejects_bad_option_type():
    with pytest.raises(ValueError):
        binomial_tree.price(S, K, R, T, SIGMA, n=10, option_type="straddle")


@pytest.mark.parametrize("option_type,expected", [("call", BS_CALL), ("put", BS_PUT)])
def test_monte_carlo_within_three_standard_errors(option_type, expected):
    price, std_err = monte_carlo.price(S, K, R, T, SIGMA, option_type=option_type, seed=42)
    assert abs(price - expected) < 3 * std_err


def test_antithetic_variates_reduce_standard_error():
    _, plain_err = monte_carlo.price(S, K, R, T, SIGMA, antithetic=False, seed=1)
    _, anti_err = monte_carlo.price(S, K, R, T, SIGMA, antithetic=True, seed=1)
    assert anti_err < plain_err


def test_monte_carlo_is_reproducible_with_seed():
    assert monte_carlo.price(S, K, R, T, SIGMA, seed=7) == monte_carlo.price(S, K, R, T, SIGMA, seed=7)
