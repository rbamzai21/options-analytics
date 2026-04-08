import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.black_scholes import call_and_put_pricing

def delta(s, k, r, t, sigma, h):
    ds = h * s
    call_up, put_up = call_and_put_pricing(s + ds, k, r, t, sigma)
    call_down, put_down = call_and_put_pricing(s - ds, k, r, t, sigma)

    c = (call_up - call_down) / (2 * ds) 
    p = (put_up - put_down) / (2 * ds)

    return c, p

def gamma(s, k, r, t, sigma, h):
    ds = h * s
    call_up, put_up = call_and_put_pricing(s + ds, k, r, t, sigma)
    call_mid, put_mid = call_and_put_pricing(s, k, r, t, sigma)
    call_down, put_down = call_and_put_pricing(s - ds, k, r, t, sigma)

    c = (call_up - (2 * call_mid) + call_down) / (ds**2) 
    p = (put_up - (2 * put_mid) + put_down) / (ds**2) 

    return c, p

def vega(s, k, r, t, sigma, h):
    call_up, put_up = call_and_put_pricing(s, k, r, t, sigma + h)
    call_down, put_down = call_and_put_pricing(s, k, r, t, sigma - h)

    c = (call_up - call_down) / (2 * h) 
    p = (put_up - put_down) / (2 * h)

    return c, p

def theta(s, k, r, t, sigma):
    dt = 1/365

    call_up, put_up = call_and_put_pricing(s, k, r, t + dt, sigma)
    call_down, put_down = call_and_put_pricing(s, k, r, t - dt, sigma)

    c = (call_up - call_down) / (2 * dt) 
    p = (put_up - put_down) / (2 * dt)

    return c, p

def rho(s, k, r, t, sigma, h):
    call_up, put_up = call_and_put_pricing(s, k, r + h, t, sigma)
    call_down, put_down = call_and_put_pricing(s, k, r - h, t, sigma)

    c = (call_up - call_down) / (2 * h) 
    p = (put_up - put_down) / (2 * h)

    return c, p

def compute_all(s, k , r, t, sigma, h=0.01):
    if t == 0:
        return {
            "call": {"delta": 1.0 if s > k else 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0},
            "put": {"delta": -1.0 if s < k else 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
        }
        
    delta_call, delta_put = delta(s, k , r, t, sigma, h)
    gamma_call, gamma_put = gamma(s, k, r, t, sigma, h)
    vega_call, vega_put = vega(s, k, r, t, sigma, h)
    theta_call, theta_put = theta(s, k, r, t, sigma)
    rho_call, rho_put = rho(s, k, r, t, sigma, h)
        
    return {
        "call": {
            "delta": delta_call,
            "gamma": gamma_call, 
            "vega": vega_call,
            "theta": theta_call, 
            "rho": rho_call
        }, 
        "put": {
            "delta": delta_put, 
            "gamma": gamma_put, 
            "vega": vega_put,  
            "theta": theta_put, 
            "rho": rho_put
        }
    }

