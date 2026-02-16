from .data_models import StockAnalysisInput
import json
import os
from typing import TypedDict

from openai import OpenAI

try:
    from langgraph.graph import END, StateGraph
except ImportError:
    END = None
    StateGraph = None


def _clamp(value: float, min_value: float = 0.0, max_value: float = 100.0) -> float:
    return max(min_value, min(max_value, value))


def _extract_json_object(text: str) -> dict:
    if not text:
        return {}

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}

    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {}


def _llm_strategy_overlay(technical_result: dict, news_result: dict, base_score: float) -> dict:
    enabled = os.getenv("USE_LLM_DECISION", "false").lower() == "true"
    api_key = os.getenv("OPENAI_API_KEY")
    if not enabled or not api_key:
        return {}

    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    prompt = (
        "You are a stock strategy reviewer. "
        "Given technical/news signals and base score, output strict JSON with keys: "
        "score_adjustment (number between -10 and 10), strategy_label (string), confidence (High|Medium|Low), "
        "short_reason (string, max 20 words). "
        "Do not include markdown.\n"
        f"technical_score={technical_result['score']} technical_outcome={technical_result['outcome']}\n"
        f"news_score={news_result['score']} news_outcome={news_result['outcome']}\n"
        f"base_strategy_score={base_score:.2f}"
    )

    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=model,
            input=prompt,
            temperature=0,
            max_output_tokens=220,
        )
        text = getattr(response, "output_text", "") or ""
        payload = _extract_json_object(text)
        if not payload:
            return {}

        adjustment = float(payload.get("score_adjustment", 0.0))
        adjustment = max(-10.0, min(10.0, adjustment))
        strategy_label = str(payload.get("strategy_label", "")).strip()
        confidence = str(payload.get("confidence", "")).strip()
        short_reason = str(payload.get("short_reason", "")).strip()

        return {
            "score_adjustment": adjustment,
            "strategy_label": strategy_label,
            "confidence": confidence,
            "short_reason": short_reason,
            "model": model,
        }
    except Exception:
        return {}


def technical_agent(input_data: StockAnalysisInput) -> dict:
    technical = input_data.technical
    sma5 = technical.sma5
    sma10 = technical.sma10

    if abs(sma10) < 1e-9:
        trend_strength = 0.0
    else:
        trend_strength = (sma5 - sma10) / abs(sma10)

    momentum = technical.momentum_20d if technical.momentum_20d is not None else trend_strength
    volatility = technical.volatility_20d if technical.volatility_20d is not None else abs(trend_strength) * 0.20

    trend_score = _clamp(50.0 + trend_strength * 600.0)
    momentum_score = _clamp(50.0 + momentum * 1200.0)
    volatility_score = _clamp(80.0 - volatility * 250.0)
    score = _clamp((0.40 * trend_score) + (0.35 * momentum_score) + (0.25 * volatility_score))

    if score >= 65:
        outcome = "Bullish"
    elif score <= 40:
        outcome = "Bearish"
    else:
        outcome = "Neutral"

    summary = (
        f"Trend={trend_strength:.2%}, Momentum20d={momentum:.2%}, "
        f"Volatility20d={volatility:.2%}, Signal={outcome}."
    )
    return {
        "agent": "technical",
        "score": round(score, 2),
        "outcome": outcome,
        "summary": summary,
    }


def news_agent(input_data: StockAnalysisInput) -> dict:
    headlines = input_data.company_news.headlines or []
    positive_words = {
        "beat", "surge", "growth", "profit", "upgrade", "strong", "record", "expand",
        "partnership", "approval", "bullish", "win"
    }
    negative_words = {
        "miss", "decline", "drop", "downgrade", "loss", "lawsuit", "weak", "cut",
        "probe", "delay", "bearish", "risk"
    }

    positive_hits = 0
    negative_hits = 0
    for headline in headlines:
        words = set(headline.lower().split())
        positive_hits += sum(1 for word in positive_words if word in words)
        negative_hits += sum(1 for word in negative_words if word in words)

    total_hits = positive_hits + negative_hits
    if total_hits == 0:
        score = 50.0
    else:
        sentiment = (positive_hits - negative_hits) / total_hits
        score = _clamp(50.0 + (sentiment * 40.0))

    if score >= 60:
        outcome = "Positive"
    elif score <= 40:
        outcome = "Negative"
    else:
        outcome = "Mixed"

    summary = f"Headlines={len(headlines)}, PositiveHits={positive_hits}, NegativeHits={negative_hits}, Sentiment={outcome}."
    return {
        "agent": "news",
        "score": round(score, 2),
        "outcome": outcome,
        "summary": summary,
    }


def strategy_agent(input_data: StockAnalysisInput, technical_result: dict, news_result: dict) -> dict:
    score = (0.65 * technical_result["score"]) + (0.35 * news_result["score"])

    if input_data.interest_rate.change == "increase":
        score -= 5.0
    elif input_data.interest_rate.change == "decrease":
        score += 5.0

    if input_data.financials.cash_reserves > input_data.financials.total_debt:
        score += 4.0
    elif input_data.financials.total_debt > (2 * max(input_data.financials.cash_reserves, 1.0)):
        score -= 6.0

    if input_data.financials.operating_cash_flow > 0:
        score += 3.0
    else:
        score -= 3.0

    score += min(5.0, len(input_data.competitor.bankruptcies) * 1.5)
    score = _clamp(score)

    llm_overlay = _llm_strategy_overlay(technical_result, news_result, score)
    if llm_overlay:
        score = _clamp(score + llm_overlay.get("score_adjustment", 0.0))

    if score >= 70:
        strategy = "Aggressive Accumulate"
    elif score >= 55:
        strategy = "Accumulate on Dips"
    elif score >= 45:
        strategy = "Hold and Monitor"
    elif score >= 30:
        strategy = "Reduce Exposure"
    else:
        strategy = "Defensive Exit"

    summary = f"Strategy={strategy}, blended from technical/news with macro-financial adjustments."
    if llm_overlay:
        llm_reason = llm_overlay.get("short_reason") or "LLM overlay applied."
        summary = (
            f"{summary} LLM({llm_overlay.get('model')}) adjustment={llm_overlay.get('score_adjustment', 0.0):+.2f}. "
            f"Reason={llm_reason}"
        )
    return {
        "agent": "strategy",
        "score": round(score, 2),
        "outcome": strategy,
        "summary": summary,
        "llm_used": bool(llm_overlay),
        "llm_confidence": llm_overlay.get("confidence", "") if llm_overlay else "",
    }


def broker_agent(input_data: StockAnalysisInput, strategy_result: dict) -> dict:
    score = strategy_result["score"]
    volatility = input_data.technical.volatility_20d if input_data.technical.volatility_20d is not None else 0.20

    if score >= 65:
        action = "Buy"
        base_allocation = 0.35
    elif score >= 45:
        action = "Hold"
        base_allocation = 0.20
    else:
        action = "Sell"
        base_allocation = 0.05

    risk_adjusted_allocation = max(0.05, min(0.50, base_allocation * (1 - min(volatility, 0.6))))
    stop_loss = max(0.03, min(0.15, 0.08 + (volatility * 0.4)))
    broker_score = _clamp((0.75 * score) + (0.25 * (100 - (volatility * 100))))

    summary = (
        f"Action={action}, Allocation={risk_adjusted_allocation:.0%}, StopLoss={stop_loss:.0%}."
    )
    return {
        "agent": "broker",
        "score": round(broker_score, 2),
        "outcome": action,
        "summary": summary,
    }


def _finalize_result(technical_result: dict, news_result: dict, strategy_result: dict, broker_result: dict) -> dict:
    overall_score = _clamp(
        (0.35 * technical_result["score"])
        + (0.20 * news_result["score"])
        + (0.35 * strategy_result["score"])
        + (0.10 * broker_result["score"])
    )

    if overall_score >= 65:
        verdict = "Rise"
        outlook = "Optimistic"
    elif overall_score <= 40:
        verdict = "Fall"
        outlook = "Defensive"
    else:
        verdict = "Sideways"
        outlook = "Cautious"

    distance = abs(overall_score - 50)
    if distance >= 20:
        confidence = "High"
    elif distance >= 10:
        confidence = "Medium"
    else:
        confidence = "Low"

    reasoning = (
        f"Scores -> Technical: {technical_result['score']}, News: {news_result['score']}, "
        f"Strategy: {strategy_result['score']}, Broker: {broker_result['score']}, "
        f"Overall: {overall_score:.2f}. "
        f"{technical_result['summary']} {news_result['summary']} {strategy_result['summary']} {broker_result['summary']}"
    )

    return {
        "technical": technical_result,
        "news": news_result,
        "strategy": strategy_result,
        "broker": broker_result,
        "overall_score": round(overall_score, 2),
        "verdict": verdict,
        "reasoning": reasoning,
        "confidence": confidence,
        "investor_outlook": outlook,
    }


def _run_agent_pipeline_direct(input_data: StockAnalysisInput) -> dict:
    technical_result = technical_agent(input_data)
    news_result = news_agent(input_data)
    strategy_result = strategy_agent(input_data, technical_result, news_result)
    broker_result = broker_agent(input_data, strategy_result)
    return _finalize_result(technical_result, news_result, strategy_result, broker_result)


class AgentPipelineState(TypedDict):
    input_data: StockAnalysisInput
    technical: dict
    news: dict
    strategy: dict
    broker: dict
    output: dict


def _technical_node(state: AgentPipelineState) -> AgentPipelineState:
    state["technical"] = technical_agent(state["input_data"])
    return state


def _news_node(state: AgentPipelineState) -> AgentPipelineState:
    state["news"] = news_agent(state["input_data"])
    return state


def _strategy_node(state: AgentPipelineState) -> AgentPipelineState:
    state["strategy"] = strategy_agent(state["input_data"], state["technical"], state["news"])
    return state


def _broker_node(state: AgentPipelineState) -> AgentPipelineState:
    state["broker"] = broker_agent(state["input_data"], state["strategy"])
    return state


def _finalize_node(state: AgentPipelineState) -> AgentPipelineState:
    state["output"] = _finalize_result(
        state["technical"],
        state["news"],
        state["strategy"],
        state["broker"],
    )
    return state


_PIPELINE_GRAPH = None


def _get_pipeline_graph():
    global _PIPELINE_GRAPH

    if _PIPELINE_GRAPH is not None:
        return _PIPELINE_GRAPH
    if StateGraph is None:
        return None

    workflow = StateGraph(AgentPipelineState)
    workflow.add_node("technical", _technical_node)
    workflow.add_node("news", _news_node)
    workflow.add_node("strategy", _strategy_node)
    workflow.add_node("broker", _broker_node)
    workflow.add_node("finalize", _finalize_node)

    workflow.set_entry_point("technical")
    workflow.add_edge("technical", "news")
    workflow.add_edge("news", "strategy")
    workflow.add_edge("strategy", "broker")
    workflow.add_edge("broker", "finalize")
    workflow.add_edge("finalize", END)

    _PIPELINE_GRAPH = workflow.compile()
    return _PIPELINE_GRAPH


def run_agent_pipeline(input_data: StockAnalysisInput) -> dict:
    graph = _get_pipeline_graph()
    if graph is None:
        return _run_agent_pipeline_direct(input_data)

    result_state = graph.invoke(
        {
            "input_data": input_data,
            "technical": {},
            "news": {},
            "strategy": {},
            "broker": {},
            "output": {},
        }
    )
    output = result_state.get("output", {})
    if output:
        return output
    return _run_agent_pipeline_direct(input_data)


def analyze_stock(input_data: StockAnalysisInput):
    result = run_agent_pipeline(input_data)
    return result["verdict"], result["reasoning"], result["confidence"], result["investor_outlook"]