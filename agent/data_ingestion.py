import yfinance as yf
import os
from datetime import datetime, timedelta
from pathlib import Path
import re

import requests
import pandas as pd


SP500_SOURCE_URL = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
SP500_LOCAL_FILE = Path(__file__).resolve().parent / "data" / "sp500_constituents.csv"
SP500_PRICE_LOCAL_FILE = Path(__file__).resolve().parent / "data" / "sp500_prices_5y.csv"
SP500_FALLBACK = {
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA",
    "BRK.B", "JPM", "V", "MA", "UNH", "XOM", "LLY", "AVGO", "COST",
    "HD", "PG", "JNJ", "MRK", "ABBV", "PEP", "KO", "BAC"
}


def _normalize_ticker_for_yfinance(ticker: str) -> str:
    return ticker.upper().replace(".", "-")


def get_sp500_tickers() -> set:
    """
    Load S&P 500 tickers from local CSV first.
    Falls back to remote CSV and then a small static subset.
    """
    try:
        table = pd.read_csv(SP500_LOCAL_FILE)
        symbols = set(table["Symbol"].astype(str).str.upper().tolist())
        if symbols:
            return symbols
    except Exception:
        pass

    try:
        table = pd.read_csv(SP500_SOURCE_URL)
        symbols = set(table["Symbol"].astype(str).str.upper().tolist())
        if symbols:
            return symbols
    except Exception:
        pass

    return SP500_FALLBACK


def is_sp500_ticker(ticker: str) -> bool:
    return ticker.upper() in get_sp500_tickers()


def _period_to_days(period: str) -> int:
    match = re.fullmatch(r"(\d+)(d|mo|y)", period)
    if not match:
        return 30

    amount = int(match.group(1))
    unit = match.group(2)
    if unit == "d":
        return amount
    if unit == "mo":
        return amount * 30
    if unit == "y":
        return amount * 365
    return 30


def fetch_local_price_data(ticker: str, period: str = "1mo", interval: str = "1d"):
    """
    Fetch historical price data for a ticker from local S&P 500 CSV.
    """
    if interval != "1d":
        raise ValueError("Local price data currently supports only interval='1d'.")

    if not SP500_PRICE_LOCAL_FILE.exists():
        return pd.DataFrame()

    symbol = ticker.upper()
    data = pd.read_csv(SP500_PRICE_LOCAL_FILE)
    if data.empty or "ticker" not in data.columns or "date" not in data.columns:
        return pd.DataFrame()

    data = data[data["ticker"].astype(str).str.upper() == symbol].copy()
    if data.empty:
        return pd.DataFrame()

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data.dropna(subset=["date"], inplace=True)
    if data.empty:
        return pd.DataFrame()

    max_date = data["date"].max()
    days = _period_to_days(period)
    start_date = max_date - pd.Timedelta(days=days)
    data = data[data["date"] >= start_date].copy()

    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    for source_col, target_col in rename_map.items():
        if source_col in data.columns:
            data[target_col] = pd.to_numeric(data[source_col], errors="coerce")

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        return pd.DataFrame()

    data = data[["date", *required_cols]].dropna(subset=["Close"]) 
    data.sort_values("date", inplace=True)
    data.set_index("date", inplace=True)
    data.index.name = "Date"
    return data


def fetch_price_data(ticker: str, period: str = "1mo", interval: str = "1d", source: str = "auto"):
    """
    Fetch historical price data for a given ticker.
    source: 'auto' (local then live), 'local', or 'live'.
    """
    source_value = (source or os.getenv("PRICE_DATA_SOURCE", "auto")).lower()

    if source_value in {"auto", "local"}:
        local_data = fetch_local_price_data(ticker, period=period, interval=interval)
        if not local_data.empty:
            return local_data
        if source_value == "local":
            return local_data

    normalized = _normalize_ticker_for_yfinance(ticker)
    data = yf.download(normalized, period=period, interval=interval, auto_adjust=True)
    return data

def calculate_sma(data, window: int):
    """
    Calculate Simple Moving Average (SMA) for the given window.
    """
    sma_value = data['Close'].rolling(window=window).mean().iloc[-1]
    if hasattr(sma_value, "iloc"):
        return float(sma_value.iloc[0])
    return float(sma_value)


def get_technical_indicators(ticker: str, period: str = "6mo", interval: str = "1d", source: str = "auto"):
    """
    Fetch price data and compute trend, momentum, and volatility indicators.
    Returns a dict with SMA-5, SMA-10, 20-day momentum, and annualized 20-day volatility.
    """
    data = fetch_price_data(ticker, period=period, interval=interval, source=source)
    if data is None or data.empty:
        return {
            "sma5": 0.0,
            "sma10": 0.0,
            "momentum_20d": 0.0,
            "volatility_20d": 0.0,
        }

    close_data = data["Close"]
    if isinstance(close_data, pd.DataFrame):
        close_data = close_data.iloc[:, 0]
    close = pd.to_numeric(close_data, errors="coerce").dropna()
    if close.empty:
        return {
            "sma5": 0.0,
            "sma10": 0.0,
            "momentum_20d": 0.0,
            "volatility_20d": 0.0,
        }

    sma5 = calculate_sma(data, 5)
    sma10 = calculate_sma(data, 10)

    if len(close) > 20:
        momentum_20d = float((close.iloc[-1] / close.iloc[-21]) - 1)
    elif len(close) > 1:
        momentum_20d = float((close.iloc[-1] / close.iloc[0]) - 1)
    else:
        momentum_20d = 0.0

    returns = close.pct_change().dropna()
    rolling_returns = returns.tail(20) if len(returns) >= 20 else returns
    if rolling_returns.empty:
        volatility_20d = 0.0
    else:
        volatility_20d = float(rolling_returns.std() * (252 ** 0.5))

    return {
        "sma5": float(sma5),
        "sma10": float(sma10),
        "momentum_20d": momentum_20d,
        "volatility_20d": volatility_20d,
    }


def get_technical_trend(ticker: str, source: str = "auto"):
    """
    Fetch price data and calculate SMA-5 and SMA-10 for the ticker.
    Returns a dict with SMA values.
    """
    indicators = get_technical_indicators(ticker, source=source)
    return {
        "sma5": indicators["sma5"],
        "sma10": indicators["sma10"]
    }


def get_company_news(ticker: str, days_back: int = 7, limit: int = 5):
    """
    Fetch recent company news headlines from Finnhub.
    Requires FINNHUB_API_KEY environment variable.
    Returns a list of headline strings.
    """
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        return []

    to_date = datetime.utcnow().date()
    from_date = to_date - timedelta(days=days_back)

    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": ticker,
        "from": from_date.isoformat(),
        "to": to_date.isoformat(),
        "token": api_key,
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        items = response.json()
    except (requests.RequestException, ValueError):
        return []

    if not isinstance(items, list):
        return []

    headlines = [item.get("headline", "") for item in items if item.get("headline")]
    return headlines[:limit]