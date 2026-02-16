import unittest
from agent.analysis import analyze_stock
from agent.data_models import *

class TestAnalysis(unittest.TestCase):
    def test_placeholder(self):
        input_data = StockAnalysisInput(
            technical=TechnicalTrend(sma5=100, sma10=100),
            company_news=CompanyNews(headlines=[]),
            country_news=CountryNews(summary="", domestic=True),
            interest_rate=InterestRate(rate=0, change="no_change"),
            financials=FinancialHealth(total_debt=0, cash_reserves=0, operating_cash_flow=0),
            competitor=CompetitorStatus(bankruptcies=[])
        )
        verdict = analyze_stock(input_data)
        self.assertEqual(verdict[0], "Neutral")

if __name__ == "__main__":
    unittest.main()