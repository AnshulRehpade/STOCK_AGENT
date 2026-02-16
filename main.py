from agent.data_models import *
from agent.analysis import run_agent_pipeline
from agent.verdict import format_verdict
from agent.data_ingestion import get_technical_indicators, get_company_news, is_sp500_ticker
from agent.stock_graph import run_market_simulation_graph, run_stock_agent
import os
from dotenv import load_dotenv


load_dotenv()


def main():
    ticker = os.getenv("STOCK_TICKER", "AAPL").upper()
    price_data_source = os.getenv("PRICE_DATA_SOURCE", "auto")
    if not is_sp500_ticker(ticker):
        raise ValueError(f"Ticker {ticker} is not in the S&P 500 universe.")
    indicators = get_technical_indicators(ticker, source=price_data_source)
    company_headlines = get_company_news(ticker)
    input_data = StockAnalysisInput(
        technical=TechnicalTrend(
            sma5=indicators["sma5"],
            sma10=indicators["sma10"],
            momentum_20d=indicators["momentum_20d"],
            volatility_20d=indicators["volatility_20d"],
        ),
        company_news=CompanyNews(headlines=company_headlines),
        country_news=CountryNews(summary="", domestic=True),  # Fill later
        interest_rate=InterestRate(rate=0, change="no_change"),  # Fill later
        financials=FinancialHealth(total_debt=0, cash_reserves=0, operating_cash_flow=0),  # Fill later
        competitor=CompetitorStatus(bankruptcies=[])  # Fill later
    )
    pipeline = run_agent_pipeline(input_data)
    verdict = (
        pipeline["verdict"],
        pipeline["reasoning"],
        pipeline["confidence"],
        pipeline["investor_outlook"],
    )
    print(format_verdict(*verdict))

    print("📌 Agent Scorecard")
    print("-" * 46)
    print(f"Technical : {pipeline['technical']['score']:>6.2f} | {pipeline['technical']['outcome']}")
    print(f"News      : {pipeline['news']['score']:>6.2f} | {pipeline['news']['outcome']}")
    print(f"Strategy  : {pipeline['strategy']['score']:>6.2f} | {pipeline['strategy']['outcome']}")
    print(f"Broker    : {pipeline['broker']['score']:>6.2f} | {pipeline['broker']['outcome']}")
    print("-" * 46)
    print(f"Overall   : {pipeline['overall_score']:>6.2f}")

if __name__ == "__main__":
    main()

    simulation_result = run_market_simulation_graph(cycles=120)
    print("Best Strategy by Agent:", simulation_result["best_strategy_by_agent"])
    print("Reward Improvement by Agent:", simulation_result["reward_improvement_by_agent"])

    query = "Analyze AAPL's short-term trend using technical indicators."
    trend_result = run_stock_agent(query)
    print("Trend Tool Output:", trend_result)