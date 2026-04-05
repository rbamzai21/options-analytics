from scipy.stats import norm
import sys 
import os
import math
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.black_scholes import compute_d1_d2

def delta(s, k, r, t, sigma):
    d1, d2 = compute_d1_d2(s, k, r, t, sigma)

    c = norm.cdf(d1)
    p = norm.cdf(d1) - 1

    return c, p

def gamma(s, k, r, t, sigma):
    d1, _ = compute_d1_d2(s, k, r, t, sigma)

    roc_delta = norm.pdf(d1) / (s * sigma * math.sqrt(t))

    return roc_delta

def vega(s, k, r, t, sigma):
    d1, _ = compute_d1_d2(s, k, r, t, sigma)

    volatility = s * norm.pdf(d1) * math.sqrt(t) / 100

    return volatility

def theta(s, k, r, t, sigma):
    d1, d2 = compute_d1_d2(s, k, r, t, sigma)

    c = (-(s * norm.pdf(d1) * sigma) / (2 * math.sqrt(t)) - (r * k * math.exp(-r * t) * norm.cdf(d2))) / 365
    p = (-(s * norm.pdf(d1) * sigma) / (2 * math.sqrt(t)) + (r * k * math.exp(-r * t) * norm.cdf(-d2))) / 365
    
    return c, p

def rho(s, k, r, t, sigma):
    _, d2 = compute_d1_d2(s, k, r, t, sigma)

    c = k * t * math.exp(-r * t) * norm.cdf(d2)
    p = -k * t * math.exp(-r * t) * norm.cdf(-d2)

    return c, p


def compute_all(s, k , r, t, sigma) -> dict:
    if t == 0:
        return {
            "call": {"delta": 1.0 if s > k else 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0},
            "put": {"delta": -1.0 if s < k else 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
        }
    call_delta, put_delta = delta(s, k, r, t, sigma)
    call_theta, put_theta = theta(s, k, r, t, sigma)
    call_rho, put_rho = rho(s, k, r, t, sigma)
    shared_gamma = gamma(s, k, r, t, sigma)
    shared_vega = vega(s, k, r, t, sigma)

    return {
        "call": {
            "delta": call_delta,
            "gamma": shared_gamma, 
            "vega": shared_vega,
            "theta": call_theta, 
            "rho": call_rho
        }, 
        "put": {
            "delta": put_delta,
            "gamma": shared_gamma, 
            "vega": shared_vega,
            "theta": put_theta, 
            "rho": put_rho
        }
    }


def main():
    greeks = compute_all(100, 100, 0.05, 1.0, 0.2)
    print(json.dumps(greeks, indent=2))

if __name__ == "__main__":
    main()