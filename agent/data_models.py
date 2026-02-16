from dataclasses import dataclass
from typing import List, Optional

@dataclass
class TechnicalTrend:
    sma5: float
    sma10: float

@dataclass
class CompanyNews:
    headlines: List[str]

@dataclass
class CountryNews:
    summary: str
    domestic: bool

@dataclass
class InterestRate:
    rate: float
    change: str  # "increase", "decrease", "no_change"

@dataclass
class FinancialHealth:
    total_debt: float
    cash_reserves: float
    operating_cash_flow: float

@dataclass
class CompetitorStatus:
    bankruptcies: List[str]
    notes: Optional[str] = None

@dataclass
class StockAnalysisInput:
    technical: TechnicalTrend
    company_news: CompanyNews
    country_news: CountryNews
    interest_rate: InterestRate
    financials: FinancialHealth
    competitor: CompetitorStatus


@dataclass
class MarketConfig:
    base_demand: float = 1000.0
    supply: float = 950.0
    demand_volatility: float = 0.08
    customer_price_sensitivity: float = 0.6
    customer_quality_preference: float = 0.5
    customer_engagement_bias: float = 0.5
    competitor_intensity: float = 0.4


@dataclass
class MarketState:
    demand: float
    supply: float
    average_price: float
    customer_quality_preference: float
    engagement_potential: float


@dataclass
class SimulationKPI:
    conversion_rate: float
    roi: float
    market_share: float


@dataclass
class AgentProfile:
    name: str
    strategies: List[str]
    epsilon: float = 0.15
    learning_rate: float = 0.25
    discount_factor: float = 0.1


@dataclass
class EpisodeResult:
    agent_name: str
    cycle: int
    action: str
    reward: float
    kpi: SimulationKPI


@dataclass
class SimulationSummary:
    best_strategy_by_agent: dict
    average_reward_by_agent: dict
    reward_improvement_by_agent: dict
    results: List[EpisodeResult]