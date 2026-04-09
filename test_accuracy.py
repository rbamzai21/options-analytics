import sys, os
sys.path.append(os.path.abspath("."))

from models.black_scholes import call_and_put_pricing
from models import binomial_tree, monte_carlo
from greeks.analytical import compute_all
import math
import json

S, K, r, T, sigma = 100.0, 100.0, 0.05, 1.0, 0.2

# --- Black-Scholes ---
call, put = call_and_put_pricing(S, K, r, T, sigma)
print(f"BS Call: {call:.4f} (expected ~10.4506)")
print(f"BS Put:  {put:.4f}  (expected ~5.5735)")

# --- Put-Call Parity ---
parity = abs((call - put) - (S - K * math.exp(-r * T)))
print(f"Parity error: {parity:.2e} (should be ~0)")

# --- Binomial Tree ---
bt_call = binomial_tree.price(S, K, r, T, sigma, n=1000, option_type="call")
bt_put  = binomial_tree.price(S, K, r, T, sigma, n=1000, option_type="put")
print(f"BT Call: {bt_call:.4f} (expected ~10.4506)")
print(f"BT Put:  {bt_put:.4f}  (expected ~5.5735)")

# --- Monte Carlo ---
mc_call, mc_call_err = monte_carlo.price(S, K, r, T, sigma, option_type="call")
mc_put,  mc_put_err  = monte_carlo.price(S, K, r, T, sigma, option_type="put")
print(f"MC Call: {mc_call:.4f} ± {mc_call_err:.4f} (expected ~10.4506)")
print(f"MC Put:  {mc_put:.4f}  ± {mc_put_err:.4f}  (expected ~5.5735)")

# --- Greeks ---
greeks = compute_all(S, K, r, T, sigma)
print("\nGreeks:")
print(json.dumps(greeks, indent=2))
print("Expected call delta ~0.6368, gamma ~0.0188, vega ~0.375, theta ~-0.0176, rho ~53.232")