import math

from scipy.stats import norm

from models.black_scholes import call_and_put_pricing, compute_d1_d2


def _bs_price(s, k, r, t, sigma, option_type):
    idx = 0 if option_type == "call" else 1
    return call_and_put_pricing(s, k, r, t, sigma)[idx]


def _raw_vega(s, k, r, t, sigma):
    # dPrice/dSigma per 1.0 change in sigma (not per 1%), as Newton-Raphson needs
    d1, _ = compute_d1_d2(s, k, r, t, sigma)
    return s * norm.pdf(d1) * math.sqrt(t)


def no_arbitrage_bounds(s, k, r, t, option_type="call"):
    """Return (lower, upper) price bounds any arbitrage-free European option must satisfy."""
    pv_k = k * math.exp(-r * t)
    if option_type.lower() == "call":
        return max(s - pv_k, 0.0), s
    return max(pv_k - s, 0.0), pv_k


def within_no_arbitrage_bounds(market_price, s, k, r, t, option_type="call"):
    lower, upper = no_arbitrage_bounds(s, k, r, t, option_type)
    return lower < market_price < upper


def implied_vol_newton(market_price, s, k, r, t, option_type="call",
                       initial_guess=0.2, max_iterations=100, tolerance=1e-6):
    sigma = initial_guess
    option_type = option_type.lower()

    for _ in range(max_iterations):
        diff = _bs_price(s, k, r, t, sigma, option_type) - market_price

        if abs(diff) < tolerance:
            return sigma, True

        vega_val = _raw_vega(s, k, r, t, sigma)

        # Vega guard: a flat price curve makes the Newton step blow up
        if vega_val < 1e-8:
            return sigma, False

        sigma = sigma - diff / vega_val

        # A step outside a sensible range means Newton is diverging
        if not (1e-6 < sigma < 10.0):
            return sigma, False

    return sigma, False


def implied_vol_bisection(market_price, s, k, r, t, option_type="call",
                          sigma_low=1e-6, sigma_high=10.0, tolerance=1e-6):
    option_type = option_type.lower()

    # If the price can't be reached anywhere in the bracket, there is no solution
    price_low = _bs_price(s, k, r, t, sigma_low, option_type)
    price_high = _bs_price(s, k, r, t, sigma_high, option_type)
    if not (price_low <= market_price <= price_high):
        return float("nan"), False

    while (sigma_high - sigma_low) / 2 > tolerance:
        m = (sigma_low + sigma_high) / 2
        price = _bs_price(s, k, r, t, m, option_type)

        if price > market_price:
            sigma_high = m
        elif price < market_price:
            sigma_low = m
        else:
            return m, True

    return (sigma_low + sigma_high) / 2, True


def solve_iv(market_price, s, k, r, t, option_type="call"):
    """Implied vol with bounds checking, Newton-Raphson, and a bisection fallback.

    Returns (iv, converged). Prices outside no-arbitrage bounds return (nan, False)
    without running either solver.
    """
    if t <= 0 or not within_no_arbitrage_bounds(market_price, s, k, r, t, option_type):
        return float("nan"), False

    iv, converged = implied_vol_newton(market_price, s, k, r, t, option_type)
    if converged:
        return iv, True

    return implied_vol_bisection(market_price, s, k, r, t, option_type)
