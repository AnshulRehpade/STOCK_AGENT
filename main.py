from agent.data_models import *
from agent.analysis import analyze_stock
from agent.verdict import format_verdict
from agent.data_ingestion import get_technical_trend
from agent.stock_graph import run_market_simulation_graph, run_stock_agent


def main():
    ticker = "AAPL"  # Example ticker
    trend = get_technical_trend(ticker)
    input_data = StockAnalysisInput(
        technical=TechnicalTrend(sma5=trend["sma5"], sma10=trend["sma10"]),
        company_news=CompanyNews(headlines=[]),  # Fill later
        country_news=CountryNews(summary="", domestic=True),  # Fill later
        interest_rate=InterestRate(rate=0, change="no_change"),  # Fill later
        financials=FinancialHealth(total_debt=0, cash_reserves=0, operating_cash_flow=0),  # Fill later
        competitor=CompetitorStatus(bankruptcies=[])  # Fill later
    )
    verdict = analyze_stock(input_data)
    print(format_verdict(*verdict))

if __name__ == "__main__":
    simulation_result = run_market_simulation_graph(cycles=120)
    print("Best Strategy by Agent:", simulation_result["best_strategy_by_agent"])
    print("Reward Improvement by Agent:", simulation_result["reward_improvement_by_agent"])

    query = "Analyze AAPL's short-term trend using technical indicators."
    trend_result = run_stock_agent(query)
    print("Trend Tool Output:", trend_result)