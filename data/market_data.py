from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

MARKET_TZ = ZoneInfo("America/New_York")
SECONDS_PER_YEAR = 365 * 24 * 60 * 60
DEFAULT_RISK_FREE_RATE = 0.04


def compute_time_to_expiry(expiry: str, now: datetime | None = None) -> float:
    # Convert "YYYY-MM-DD" to fractional years from now.
    # US equity options stop trading at 4:00 PM Eastern on the expiry date.
    expiry_dt = datetime.strptime(expiry, "%Y-%m-%d").replace(hour=16, tzinfo=MARKET_TZ)
    now = now or datetime.now(MARKET_TZ)

    return (expiry_dt - now).total_seconds() / SECONDS_PER_YEAR


def clean_chain(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["bid", "ask", "volume", "openInterest"])
    df = df[df["bid"] > 0]
    df = df[df["ask"] >= df["bid"]]
    df = df[df["volume"] >= 10]
    df = df[df["openInterest"] > 0]
    return df


def get_spot_price(ticker: str) -> float:
    # Returns current stock price for a given ticker
    stock = yf.Ticker(ticker)

    price = None
    try:
        price = stock.fast_info.get("lastPrice")
    except Exception:
        pass
    if not price:
        price = stock.info.get("regularMarketPrice") or stock.info.get("previousClose")

    if not price:
        raise ValueError(f"Could not find price for ticker '{ticker}'. It may be invalid.")
    return float(price)


def get_options_chain(ticker: str, expiry: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    # returns (calls_df, puts_df) for a given ticker and expiry date
    # expiry date format: "YYYY-MM-DD"
    stock = yf.Ticker(ticker)
    chain = stock.option_chain(expiry)

    return clean_chain(chain.calls), clean_chain(chain.puts)


def get_risk_free_rate() -> float:
    # Approximate risk-free rate from the 13-week T-bill yield (ticker "^IRX").
    # ^IRX is quoted as a percentage, so divide by 100. Falls back to a default
    # if Yahoo doesn't return a value.
    try:
        irx = yf.Ticker("^IRX")
        rate = irx.fast_info.get("lastPrice") or irx.info.get("regularMarketPrice")
        if rate:
            return float(rate) / 100
    except Exception:
        pass
    return DEFAULT_RISK_FREE_RATE
