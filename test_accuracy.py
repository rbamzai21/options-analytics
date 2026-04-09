from data.market_data import get_spot_price, get_risk_free_rate, get_options_chain
import yfinance as yf

ticker = "AAPL"
spot = get_spot_price(ticker)
r = get_risk_free_rate()

print(f"Spot: ${spot}")
print(f"Risk-free rate: {r:.4f} ({r*100:.2f}%)")

# Get first available expiry
expiry = yf.Ticker(ticker).options[1]
calls, puts = get_options_chain(ticker, expiry)
print(f"\nExpiry: {expiry}")
print(calls[["strike", "bid", "ask", "volume", "openInterest"]].head(10))