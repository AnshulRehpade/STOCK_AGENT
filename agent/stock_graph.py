from typing import TypedDict

try:
    from langgraph.graph import END, StateGraph
except ImportError:
    END = None
    StateGraph = None

from agent.market_simulation import build_default_simulation, simulation_summary_to_dict
from agent.data_ingestion import get_technical_trend, is_sp500_ticker


class MarketGraphState(TypedDict):
    cycles: int
    summary: dict


def run_stock_agent(query: str):
    query_lower = query.lower()
    if "trend" in query_lower or "sma" in query_lower:
        ticker = "AAPL"
        for token in query.split():
            clean_token = token.strip(".,!?\"'")
            if clean_token.isalpha() and 1 <= len(clean_token) <= 5 and clean_token.isupper():
                ticker = clean_token
                break
        if not is_sp500_ticker(ticker):
            return {
                "message": f"{ticker} is outside the S&P 500 universe. Please use an S&P 500 ticker symbol."
            }
        try:
            from agent.tools import technical_trend_tool

            return technical_trend_tool.invoke({"ticker": ticker})
        except Exception:
            return get_technical_trend(ticker)
    return {
        "message": "Stock tool supports technical trend queries with ticker symbols, e.g. 'Analyze AAPL trend'."
    }


def _run_simulation_node(state: MarketGraphState) -> MarketGraphState:
    simulation = build_default_simulation(seed=42)
    summary = simulation.run(cycles=state["cycles"])
    state["summary"] = simulation_summary_to_dict(summary)
    return state


def run_market_simulation_graph(cycles: int = 120) -> dict:
    if StateGraph is None:
        simulation = build_default_simulation(seed=42)
        summary = simulation.run(cycles=cycles)
        return simulation_summary_to_dict(summary)

    workflow = StateGraph(MarketGraphState)
    workflow.add_node("run_simulation", _run_simulation_node)
    workflow.set_entry_point("run_simulation")
    workflow.add_edge("run_simulation", END)

    graph = workflow.compile()
    result = graph.invoke({"cycles": cycles, "summary": {}})
    return result["summary"]