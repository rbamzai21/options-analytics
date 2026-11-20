import math
from scipy.stats import norm


def compute_d1_d2(s, k, r, t, sigma):
    d1 = (math.log(s / k) + (r + (sigma * sigma) / 2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)

    return d1, d2


def call_and_put_pricing(s, k, r, t, sigma):
    # At expiry the option is worth its intrinsic value
    if t <= 0:
        return max(s - k, 0), max(k - s, 0)

    # With zero volatility the price is the discounted forward intrinsic value
    if sigma <= 0:
        pv_k = k * math.exp(-r * t)
        return max(s - pv_k, 0), max(pv_k - s, 0)

    d1, d2 = compute_d1_d2(s, k, r, t, sigma)

    c = (s * norm.cdf(d1)) - (k * math.exp(-r * t) * norm.cdf(d2))
    p = (k * math.exp(-r * t) * norm.cdf(-d2)) - (s * norm.cdf(-d1))

    return c, p


def main():
    c1, p1 = call_and_put_pricing(100, 100, 0.05, 1.0, 0.2)
    assert abs((c1 - p1) - (100 - 100 * math.exp(-0.05 * 1.0))) < 1e-10, "Put-call parity violated!"

    c2, p2 = call_and_put_pricing(100, 110, 0.05, 1.0, 0.2)

    print(c1, p1)
    print(c2, p2)


if __name__ == "__main__":
    main()
