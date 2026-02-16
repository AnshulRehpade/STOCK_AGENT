from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

from agent.analysis import run_agent_pipeline
from agent.data_ingestion import fetch_price_data, get_company_news, get_technical_indicators, is_sp500_ticker
from agent.data_models import (
    CompanyNews,
    CompetitorStatus,
    CountryNews,
    FinancialHealth,
    InterestRate,
    StockAnalysisInput,
    TechnicalTrend,
)


app = FastAPI(title="Stock Agent Dashboard")

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")


HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Stock Agent Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; margin: 0; background: #f7f8fb; color: #1e2430; }
    .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
    .header { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; justify-content: space-between; }
    .controls { display: flex; flex-wrap: wrap; gap: 8px; }
    input, select, button { padding: 10px 12px; border-radius: 8px; border: 1px solid #d8dde6; background: #fff; }
    button { background: #1f6feb; color: #fff; border: none; cursor: pointer; }
    .grid { display: grid; grid-template-columns: repeat(6, minmax(140px, 1fr)); gap: 12px; margin-top: 16px; }
    .card { background: #fff; border-radius: 10px; padding: 12px; box-shadow: 0 2px 8px rgba(20, 29, 45, 0.06); }
    .label { font-size: 12px; color: #667085; margin-bottom: 4px; }
    .value { font-size: 18px; font-weight: 600; }
    .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px; }
    .chart-card { background: #fff; border-radius: 10px; padding: 10px; box-shadow: 0 2px 8px rgba(20, 29, 45, 0.06); }
    .reasoning { margin-top: 12px; background: #fff; border-radius: 10px; padding: 12px; box-shadow: 0 2px 8px rgba(20, 29, 45, 0.06); line-height: 1.4; }
    .headlines { margin-top: 12px; background: #fff; border-radius: 10px; padding: 12px; box-shadow: 0 2px 8px rgba(20, 29, 45, 0.06); }
    .trace { margin-top: 12px; background: #fff; border-radius: 10px; padding: 12px; box-shadow: 0 2px 8px rgba(20, 29, 45, 0.06); }
    table { width: 100%; border-collapse: collapse; margin-top: 8px; }
    th, td { text-align: left; padding: 8px; border-bottom: 1px solid #eef1f6; font-size: 14px; }
    th { color: #475467; font-weight: 600; }
    ul { margin: 8px 0 0 16px; }
    .error { color: #b42318; margin-top: 10px; }
    @media (max-width: 980px) {
      .grid { grid-template-columns: repeat(2, minmax(140px, 1fr)); }
      .charts { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h2>Stock Agent Dashboard</h2>
      <div class="controls">
        <input id="ticker" value="AAPL" maxlength="6" placeholder="Ticker" />
        <select id="source">
          <option value="auto">auto</option>
          <option value="local">local</option>
          <option value="live">live</option>
        </select>
        <select id="period">
          <option value="6mo">6mo</option>
          <option value="1y" selected>1y</option>
          <option value="5y">5y</option>
        </select>
        <button onclick="runAnalysis()">Analyze</button>
      </div>
    </div>

    <div id="error" class="error"></div>

    <div class="grid">
      <div class="card"><div class="label">Verdict</div><div id="verdict" class="value">-</div></div>
      <div class="card"><div class="label">Confidence</div><div id="confidence" class="value">-</div></div>
      <div class="card"><div class="label">Outlook</div><div id="outlook" class="value">-</div></div>
      <div class="card"><div class="label">Overall Score</div><div id="overall" class="value">-</div></div>
      <div class="card"><div class="label">Ticker</div><div id="tickerLabel" class="value">-</div></div>
      <div class="card"><div class="label">Action</div><div id="actionLabel" class="value">-</div></div>
    </div>

    <div class="charts">
      <div class="chart-card"><div id="priceChart" style="height: 360px;"></div></div>
      <div class="chart-card"><div id="scoreChart" style="height: 360px;"></div></div>
    </div>

    <div class="reasoning">
      <div class="label">Reasoning</div>
      <div id="reasoning">-</div>
    </div>

    <div class="headlines">
      <div class="label">Latest Company Headlines</div>
      <ul id="headlinesList"></ul>
    </div>

    <div class="trace">
      <div class="label">LangGraph Trace</div>
      <table>
        <thead>
          <tr>
            <th>Node</th>
            <th>Score</th>
            <th>Outcome</th>
          </tr>
        </thead>
        <tbody id="traceTableBody"></tbody>
      </table>
    </div>
  </div>

  <script>
    function toPct(value) {
      return `${(value * 100).toFixed(2)}%`;
    }

    async function runAnalysis() {
      const ticker = document.getElementById('ticker').value.trim().toUpperCase();
      const source = document.getElementById('source').value;
      const period = document.getElementById('period').value;
      const errorEl = document.getElementById('error');
      errorEl.textContent = '';

      try {
        const res = await fetch(`/api/analyze?ticker=${encodeURIComponent(ticker)}&source=${encodeURIComponent(source)}&period=${encodeURIComponent(period)}`);
        const payload = await res.json();
        if (!res.ok) {
          throw new Error(payload.detail || 'Failed to analyze');
        }

        document.getElementById('verdict').textContent = payload.pipeline.verdict;
        document.getElementById('confidence').textContent = payload.pipeline.confidence;
        document.getElementById('outlook').textContent = payload.pipeline.investor_outlook;
        document.getElementById('overall').textContent = payload.pipeline.overall_score.toFixed(2);
        document.getElementById('tickerLabel').textContent = payload.ticker;
        document.getElementById('actionLabel').textContent = payload.pipeline?.broker?.outcome || '-';
        document.getElementById('reasoning').textContent = payload.pipeline.reasoning;

        const headlinesList = document.getElementById('headlinesList');
        headlinesList.innerHTML = '';
        const headlines = payload.news_headlines || [];
        if (headlines.length === 0) {
          const li = document.createElement('li');
          li.textContent = 'No headlines available (set FINNHUB_API_KEY for live news).';
          headlinesList.appendChild(li);
        } else {
          headlines.forEach((h) => {
            const li = document.createElement('li');
            li.textContent = h;
            headlinesList.appendChild(li);
          });
        }

        const traceBody = document.getElementById('traceTableBody');
        traceBody.innerHTML = '';
        const trace = payload.graph_trace || [];
        trace.forEach((step) => {
          const tr = document.createElement('tr');

          const nodeTd = document.createElement('td');
          nodeTd.textContent = step.node;

          const scoreTd = document.createElement('td');
          scoreTd.textContent = (typeof step.score === 'number') ? step.score.toFixed(2) : '-';

          const outcomeTd = document.createElement('td');
          outcomeTd.textContent = step.outcome || '-';

          tr.appendChild(nodeTd);
          tr.appendChild(scoreTd);
          tr.appendChild(outcomeTd);
          traceBody.appendChild(tr);
        });

        const priceX = payload.price_history.map((x) => x.date);
        const priceY = payload.price_history.map((x) => x.close);
        Plotly.newPlot('priceChart', [{ x: priceX, y: priceY, type: 'scatter', mode: 'lines', name: 'Close' }], {
          title: `${payload.ticker} Close Price (${payload.period})`,
          margin: { t: 40, l: 40, r: 20, b: 40 },
        }, {displayModeBar: false});

        const labels = ['Technical', 'News', 'Strategy', 'Broker', 'Overall'];
        const values = [
          payload.pipeline.technical.score,
          payload.pipeline.news.score,
          payload.pipeline.strategy.score,
          payload.pipeline.broker.score,
          payload.pipeline.overall_score,
        ];
        Plotly.newPlot('scoreChart', [{ x: labels, y: values, type: 'bar' }], {
          title: 'Agent Scores',
          yaxis: { range: [0, 100] },
          margin: { t: 40, l: 40, r: 20, b: 40 },
        }, {displayModeBar: false});
      } catch (err) {
        errorEl.textContent = err.message;
      }
    }

    runAnalysis();
  </script>
</body>
</html>
"""


def _close_series_from_frame(data: pd.DataFrame) -> pd.Series:
    if data is None or data.empty:
        return pd.Series(dtype=float)

    close_data = data["Close"] if "Close" in data.columns else pd.Series(dtype=float)
    if isinstance(close_data, pd.DataFrame):
        close_data = close_data.iloc[:, 0]

    close_series = pd.to_numeric(close_data, errors="coerce").dropna()
    return close_series


@app.get("/", response_class=HTMLResponse)
def dashboard_home() -> str:
    return HTML_PAGE


@app.get("/api/analyze")
def analyze_ticker(
    ticker: str = Query(default="AAPL", min_length=1, max_length=6),
    source: str = Query(default="auto", pattern="^(auto|local|live)$"),
    period: str = Query(default="1y", pattern="^(1mo|3mo|6mo|1y|2y|5y)$"),
):
    symbol = ticker.strip().upper()
    if not is_sp500_ticker(symbol):
        raise HTTPException(status_code=400, detail=f"Ticker {symbol} is not in the S&P 500 universe.")

    price_data = fetch_price_data(symbol, period=period, interval="1d", source=source)
    close_series = _close_series_from_frame(price_data)
    if close_series.empty:
        raise HTTPException(status_code=404, detail=f"No price data found for {symbol}.")

    indicators = get_technical_indicators(symbol, period=period, interval="1d", source=source)
    headlines = get_company_news(symbol)

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

    pipeline = run_agent_pipeline(input_data)

    price_history = [
        {"date": idx.strftime("%Y-%m-%d"), "close": float(value)}
        for idx, value in close_series.items()
    ]

    return {
        "ticker": symbol,
        "source": source,
        "period": period,
        "indicators": indicators,
        "pipeline": pipeline,
      "graph_trace": [
        {
          "node": "technical",
          "score": pipeline["technical"]["score"],
          "outcome": pipeline["technical"]["outcome"],
        },
        {
          "node": "news",
          "score": pipeline["news"]["score"],
          "outcome": pipeline["news"]["outcome"],
        },
        {
          "node": "strategy",
          "score": pipeline["strategy"]["score"],
          "outcome": pipeline["strategy"]["outcome"],
        },
        {
          "node": "broker",
          "score": pipeline["broker"]["score"],
          "outcome": pipeline["broker"]["outcome"],
        },
        {
          "node": "finalize",
          "score": pipeline["overall_score"],
          "outcome": pipeline["verdict"],
        },
      ],
        "news_headlines": headlines,
        "price_history": price_history,
    }
