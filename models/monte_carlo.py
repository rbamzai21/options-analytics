import numpy as np
import math

def price(s, k, r, t, sigma, n_simulations=100000, option_type="call", antithetic=True):    
    option_type = option_type.lower()
    
    if antithetic:
        z = np.random.standard_normal(n_simulations // 2)
        z = np.concatenate([z, -z])
    else:
        z = np.random.standard_normal(n_simulations)
        
    st = s * np.exp((r - (sigma*sigma/2)) * t + (sigma * math.sqrt(t) * z))
    
    if option_type == "call":
        payoffs = np.maximum(st - k, 0)
    else: 
        payoffs = np.maximum(k - st, 0)    
    
    option_price = math.exp(-r * t) * np.mean(payoffs)
    std_error = np.std(payoffs) / math.sqrt(n_simulations)
        

    return option_price, std_error