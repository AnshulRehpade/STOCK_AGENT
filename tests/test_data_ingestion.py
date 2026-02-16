import unittest
import pandas as pd

from agent.data_ingestion import (
    calculate_sma,
    fetch_local_price_data,
    fetch_price_data,
    get_sp500_tickers,
    is_sp500_ticker,
)


class TestDataIngestion(unittest.TestCase):
    def test_calculate_sma_returns_float_with_single_index_columns(self):
        data = pd.DataFrame(
            {"Close": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        )

        sma = calculate_sma(data, 5)

        self.assertIsInstance(sma, float)
        self.assertEqual(sma, 8.0)

    def test_calculate_sma_returns_float_with_multiindex_columns(self):
        columns = pd.MultiIndex.from_tuples([("Close", "AAPL")], names=["Price", "Ticker"])
        data = pd.DataFrame(
            [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]],
            columns=columns,
        )

        sma = calculate_sma(data, 5)

        self.assertIsInstance(sma, float)
        self.assertEqual(sma, 8.0)

    def test_get_sp500_tickers_contains_common_symbols(self):
        tickers = get_sp500_tickers()

        self.assertIn("AAPL", tickers)
        self.assertIn("MSFT", tickers)
        self.assertIn("NVDA", tickers)
        self.assertGreaterEqual(len(tickers), 500)

    def test_is_sp500_ticker_true_for_known_symbol(self):
        self.assertTrue(is_sp500_ticker("AAPL"))

    def test_fetch_local_price_data_returns_expected_columns(self):
        data = fetch_local_price_data("AAPL", period="1mo", interval="1d")

        self.assertFalse(data.empty)
        self.assertIn("Open", data.columns)
        self.assertIn("High", data.columns)
        self.assertIn("Low", data.columns)
        self.assertIn("Close", data.columns)
        self.assertIn("Volume", data.columns)

    def test_fetch_price_data_local_source_uses_local_dataset(self):
        data = fetch_price_data("AAPL", period="1mo", interval="1d", source="local")

        self.assertFalse(data.empty)
        self.assertIn("Close", data.columns)


if __name__ == "__main__":
    unittest.main()
