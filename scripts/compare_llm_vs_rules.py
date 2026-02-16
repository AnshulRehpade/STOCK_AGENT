import os
from dotenv import load_dotenv

from agent.analysis import run_agent_pipeline
from agent.data_ingestion import get_company_news, get_technical_indicators
from agent.data_models import (
    CompanyNews,
    CompetitorStatus,
    CountryNews,
    FinancialHealth,
    InterestRate,
    StockAnalysisInput,
    TechnicalTrend,
)


def main() -> None:
    load_dotenv()

    ticker = os.getenv("STOCK_TICKER", "AAPL").upper()
    source = os.getenv("PRICE_DATA_SOURCE", "local")

    indicators = get_technical_indicators(ticker, source=source)
    headlines = get_company_news(ticker)

    input_data = StockAnalysisInput(
        technical=TechnicalTrend(
            sma5=indicators["sma5"],
            sma10=indicators["sma10"],
            momentum_20d=indicators["momentum_20d"],
            volatility_20d=indicators["volatility_20d"],
        ),
        company_news=CompanyNews(headlines=headlines),
        country_news=CountryNews(summary="", domestic=True),
        interest_rate=InterestRate(rate=0, change="no_change"),
        financials=FinancialHealth(total_debt=0, cash_reserves=0, operating_cash_flow=0),
        competitor=CompetitorStatus(bankruptcies=[]),
    )

    os.environ["USE_LLM_DECISION"] = "false"
    rule_only = run_agent_pipeline(input_data)

    os.environ["USE_LLM_DECISION"] = "true"
    llm_on = run_agent_pipeline(input_data)

    print("RULE_ONLY:")
    print("  verdict=", rule_only["verdict"])
    print("  overall=", rule_only["overall_score"])
    print("  strategy_score=", rule_only["strategy"]["score"])
    print("  strategy_outcome=", rule_only["strategy"]["outcome"])
    print("  llm_used=", rule_only["strategy"].get("llm_used"))

    print("LLM_ON:")
    print("  verdict=", llm_on["verdict"])
    print("  overall=", llm_on["overall_score"])
    print("  strategy_score=", llm_on["strategy"]["score"])
    print("  strategy_outcome=", llm_on["strategy"]["outcome"])
    print("  llm_used=", llm_on["strategy"].get("llm_used"))
    print("  llm_confidence=", llm_on["strategy"].get("llm_confidence"))

    print("DELTA:")
    print("  overall_delta=", round(llm_on["overall_score"] - rule_only["overall_score"], 4))
    print("  strategy_delta=", round(llm_on["strategy"]["score"] - rule_only["strategy"]["score"], 4))


if __name__ == "__main__":
    main()
