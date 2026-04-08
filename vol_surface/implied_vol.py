import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.black_scholes import call_and_put_pricing
from greeks.finite_difference import vega

def implied_vol_newton(market_price, s, k, r, t, option_type="call",
                       initial_guess=0.2, max_iterations=100, tolerance=1e-6):
    
    sigma = initial_guess
    option_type = option_type.lower()

    idx = 0 if option_type == "call" else 1

    for _ in range(max_iterations):
        price = call_and_put_pricing(s, k, r, t, sigma)[idx]
        vega_val = vega(s, k, r, t, sigma, 0.01)[0]

        if abs(vega_val) < 1e-10:
            return sigma, False
        
        if abs(price - market_price) < tolerance:
            return sigma, True
        
        sigma = sigma - ((price - market_price) / vega_val)
        
    
    
    return sigma, False

def implied_vol_bisection(market_price, s, k, r, t, option_type="call",
                          sigma_low=1e-6, sigma_high=10.0, tolerance=1e-6):
    
    option_type = option_type.lower()
    
    idx = 0 if option_type == "call" else 1
    
    while (sigma_high - sigma_low) / 2 > tolerance:
        m = (sigma_low + sigma_high) / 2

        price = call_and_put_pricing(s, k, r, t, m)[idx]

        if price > market_price:
            sigma_high = m
        elif price < market_price:
            sigma_low = m
        else:
            return m, True
   
    
    return (sigma_low + sigma_high) / 2, True
