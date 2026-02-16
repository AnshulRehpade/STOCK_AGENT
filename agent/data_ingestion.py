import yfinance as yf

def fetch_price_data(ticker: str, period: str = "1mo", interval: str = "1d"):
    """
    Fetch historical price data for a given ticker.
    """
    data = yf.download(ticker, period=period, interval=interval)
    return data

def calculate_sma(data, window: int):
    """
    Calculate Simple Moving Average (SMA) for the given window.
    """
    return data['Close'].rolling(window=window).mean().iloc[-1]

def get_technical_trend(ticker: str):
    """
    Fetch price data and calculate SMA-5 and SMA-10 for the ticker.
    Returns a dict with SMA values.
    """
    data = fetch_price_data(ticker)
    sma5 = calculate_sma(data, 5)
    sma10 = calculate_sma(data, 10)
    return {
        "sma5": sma5,
        "sma10": sma10
    }