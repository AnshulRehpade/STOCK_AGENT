import unittest
from agent.analysis import analyze_stock, run_agent_pipeline
from agent.data_models import *

class TestAnalysis(unittest.TestCase):
    def test_analyze_stock_returns_structured_verdict(self):
        input_data = StockAnalysisInput(
            technical=TechnicalTrend(sma5=105, sma10=100, momentum_20d=0.03, volatility_20d=0.22),
            company_news=CompanyNews(headlines=[]),
            country_news=CountryNews(summary="", domestic=True),
            interest_rate=InterestRate(rate=0, change="no_change"),
            financials=FinancialHealth(total_debt=0, cash_reserves=0, operating_cash_flow=0),
            competitor=CompetitorStatus(bankruptcies=[])
        )
        verdict = analyze_stock(input_data)
        self.assertEqual(len(verdict), 4)
        self.assertIn(verdict[0], {"Rise", "Sideways", "Fall"})
        self.assertIn(verdict[2], {"High", "Medium", "Low"})

    def test_run_agent_pipeline_contains_agent_scores(self):
        input_data = StockAnalysisInput(
            technical=TechnicalTrend(sma5=110, sma10=100, momentum_20d=0.05, volatility_20d=0.18),
            company_news=CompanyNews(headlines=["Company reports record profit growth" ]),
            country_news=CountryNews(summary="", domestic=True),
            interest_rate=InterestRate(rate=5, change="no_change"),
            financials=FinancialHealth(total_debt=10, cash_reserves=20, operating_cash_flow=5),
            competitor=CompetitorStatus(bankruptcies=["RivalCo"])
        )

        result = run_agent_pipeline(input_data)

        self.assertIn("technical", result)
        self.assertIn("news", result)
        self.assertIn("strategy", result)
        self.assertIn("broker", result)
        self.assertIn("overall_score", result)
        self.assertTrue(0 <= result["overall_score"] <= 100)

if __name__ == "__main__":
    unittest.main()