from pathlib import Path

import pandas as pd
import yfinance as yf

from agent.data_ingestion import get_sp500_tickers


def main() -> None:
    output_path = Path("agent/data/sp500_prices_5y.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    symbols = sorted(get_sp500_tickers())
    rows = []
    failed = []

    for symbol in symbols:
        yf_symbol = symbol.replace(".", "-")
        try:
            df = yf.download(
                yf_symbol,
                period="5y",
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            if df is None or df.empty:
                failed.append(symbol)
                continue

            if isinstance(df.columns, pd.MultiIndex):
                ticker_level = None
                for level in range(df.columns.nlevels):
                    vals = set(df.columns.get_level_values(level))
                    if yf_symbol in vals or symbol in vals:
                        ticker_level = level
                        break
                if ticker_level is not None:
                    level_values = set(df.columns.get_level_values(ticker_level))
                    use_symbol = yf_symbol if yf_symbol in level_values else symbol
                    df = df.xs(use_symbol, axis=1, level=ticker_level)

            col_map = {str(c).lower(): c for c in df.columns}
            required = ["open", "high", "low", "close", "volume"]
            if not all(col in col_map for col in required):
                failed.append(symbol)
                continue

            normalized = pd.DataFrame(
                {
                    "date": df.index,
                    "ticker": symbol,
                    "open": pd.to_numeric(df[col_map["open"]], errors="coerce"),
                    "high": pd.to_numeric(df[col_map["high"]], errors="coerce"),
                    "low": pd.to_numeric(df[col_map["low"]], errors="coerce"),
                    "close": pd.to_numeric(df[col_map["close"]], errors="coerce"),
                    "volume": pd.to_numeric(df[col_map["volume"]], errors="coerce"),
                }
            )
            normalized.dropna(subset=["open", "high", "low", "close"], inplace=True)
            rows.append(normalized)
        except Exception:
            failed.append(symbol)

    if rows:
        output = pd.concat(rows, ignore_index=True)
        output.sort_values(["ticker", "date"], inplace=True)
        output.to_csv(output_path, index=False)
        print(f"OUTPUT: {output_path.as_posix()}")
        print(f"TICKERS_TOTAL: {len(symbols)}")
        print(f"TICKERS_SUCCESS: {output['ticker'].nunique()}")
        print(f"TICKERS_FAILED: {len(set(failed))}")
        print(f"ROWS: {len(output)}")
    else:
        print("No data downloaded.")
        print(f"TICKERS_TOTAL: {len(symbols)}")
        print(f"TICKERS_FAILED: {len(set(failed))}")


if __name__ == "__main__":
    main()
