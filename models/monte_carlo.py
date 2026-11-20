import math
import numpy as np


def price(s, k, r, t, sigma, n_simulations=100000, option_type="call", antithetic=True, seed=None):
    """Price a European option by Monte Carlo simulation under GBM.

    Returns (price, standard_error). With antithetic=True, each draw Z is paired
    with -Z, and the standard error is computed from the pair averages, since
    paired paths are not independent.
    """
    option_type = option_type.lower()
    rng = np.random.default_rng(seed)
    discount = math.exp(-r * t)

    def payoff(z):
        st = s * np.exp((r - sigma * sigma / 2) * t + sigma * math.sqrt(t) * z)
        if option_type == "call":
            return np.maximum(st - k, 0)
        return np.maximum(k - st, 0)

    if antithetic:
        n_pairs = n_simulations // 2
        z = rng.standard_normal(n_pairs)
        samples = discount * (payoff(z) + payoff(-z)) / 2
    else:
        z = rng.standard_normal(n_simulations)
        samples = discount * payoff(z)

    option_price = float(np.mean(samples))
    std_error = float(np.std(samples, ddof=1) / math.sqrt(len(samples)))

    return option_price, std_error
