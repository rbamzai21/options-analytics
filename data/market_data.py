import pandas as pd
from datetime import datetime
import yfinance as yf

def compute_time_to_expiry(expiry: str) -> float:
    # convert "YYYY-MM-DD" to fractional years from today
    expiry_dt = datetime.strptime(expiry, "%Y-%m-%d")

    return (expiry_dt - datetime.today()).days / 365

def clean_chain(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["bid"] > 0]
    df = df[df["volume"] >= 10]
    df = df[df["openInterest"] > 0]
    return df.dropna(subset=["bid", "volume", "openInterest"])


def get_spot_price(ticker: str) -> float:
    # Returns current stock price for a given ticker
    stock = yf.Ticker(ticker)
    
    price = stock.info.get("regularMarketPrice") or stock.info.get("previousClose")
    
    if price is None:
        raise ValueError(f"Could not find price for ticker '{ticker}'. It may be invalid.")
    return price

def get_options_chain(ticker: str, expiry: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    # returns (calls_df, puts_df) for a given ticker and expiry date
    # expiry date format: "YYYY-MM-DD"
    stock = yf.Ticker(ticker)
    chain = stock.option_chain(expiry)
    
    return clean_chain(chain.calls), clean_chain(chain.puts)

def get_risk_free_rate() -> float:
    # returns current approximate risk-free rate
    # use the 13-week T-bill yield (ticker: "^IRX")
    # ^IRX is quoted as a percentage, so divide by 100
    irx = yf.Ticker("^IRX")
    return irx.info["regularMarketPrice"] / 100