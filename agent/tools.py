from langchain.tools import tool
from agent.data_ingestion import get_technical_trend

@tool
def technical_trend_tool(ticker: str) -> dict:
    """Fetches SMA-5 and SMA-10 for a given stock ticker."""
    return get_technical_trend(ticker)