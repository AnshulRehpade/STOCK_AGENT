import asyncio
import random
from collections import defaultdict
from dataclasses import asdict
from typing import Dict, List, Tuple

from .data_models import (
    AgentProfile,
    EpisodeResult,
    MarketConfig,
    MarketState,
    SimulationKPI,
    SimulationSummary,
)


class MarketEnvironment:
    def __init__(self, config: MarketConfig, seed: int = 42):
        self.config = config
        self.random = random.Random(seed)
        self.state = self._build_initial_state()

    def _build_initial_state(self) -> MarketState:
        return MarketState(
            demand=self.config.base_demand,
            supply=self.config.supply,
            average_price=100.0,
            customer_quality_preference=self.config.customer_quality_preference,
            engagement_potential=self.config.customer_engagement_bias,
        )

    def reset(self) -> MarketState:
        self.state = self._build_initial_state()
        return self.state

    def _mutate_market(self) -> None:
        demand_noise = 1 + self.random.uniform(
            -self.config.demand_volatility, self.config.demand_volatility
        )
        self.state.demand = max(100.0, self.state.demand * demand_noise)
        self.state.customer_quality_preference = min(
            1.0,
            max(
                0.0,
                self.state.customer_quality_preference + self.random.uniform(-0.05, 0.05),
            ),
        )
        self.state.engagement_potential = min(
            1.0,
            max(0.0, self.state.engagement_potential + self.random.uniform(-0.06, 0.06)),
        )

    def evaluate_action(self, action: str) -> Tuple[float, SimulationKPI]:
        demand_pressure = self.state.demand / max(self.state.supply, 1.0)
        competitiveness = 1.0 - (self.config.competitor_intensity * 0.3)

        if action == "aggressive_pricing":
            conversion = min(
                0.95,
                0.45
                + self.config.customer_price_sensitivity * 0.35
                + demand_pressure * 0.08,
            )
            roi = max(0.01, 0.22 + demand_pressure * 0.05 - 0.20 * competitiveness)
            market_share = min(1.0, 0.30 + conversion * 0.38)
        elif action == "value_targeting":
            conversion = min(
                0.95,
                0.35
                + self.state.customer_quality_preference * 0.42
                + demand_pressure * 0.05,
            )
            roi = max(
                0.01,
                0.28 + self.state.customer_quality_preference * 0.24 - 0.06 * competitiveness,
            )
            market_share = min(1.0, 0.26 + conversion * 0.33)
        elif action == "lead_engagement":
            conversion = min(
                0.95,
                0.31 + self.state.engagement_potential * 0.45 + demand_pressure * 0.05,
            )
            roi = max(0.01, 0.24 + self.state.engagement_potential * 0.23 - 0.10 * competitiveness)
            market_share = min(1.0, 0.22 + conversion * 0.35)
        else:
            raise ValueError(f"Unknown strategy action: {action}")

        noise = self.random.uniform(-0.02, 0.02)
        conversion = min(0.99, max(0.01, conversion + noise))
        roi = min(1.5, max(0.01, roi + noise))
        market_share = min(0.99, max(0.01, market_share + noise))

        reward = conversion * 0.35 + roi * 0.45 + market_share * 0.20
        reward *= 100

        return reward, SimulationKPI(conversion_rate=conversion, roi=roi, market_share=market_share)

    def step(self, action: str) -> Tuple[MarketState, float, SimulationKPI]:
        reward, kpi = self.evaluate_action(action)
        self._mutate_market()
        return self.state, reward, kpi


class StrategyRLAgent:
    def __init__(self, profile: AgentProfile, seed: int = 123):
        self.profile = profile
        self.q_values = defaultdict(float)
        self.action_counts = defaultdict(int)
        self.random = random.Random(seed)

    def choose_action(self) -> str:
        if self.random.random() < self.profile.epsilon:
            return self.random.choice(self.profile.strategies)
        return max(self.profile.strategies, key=lambda action: self.q_values[action])

    def learn(self, action: str, reward: float) -> None:
        current_q = self.q_values[action]
        best_next_q = max(self.q_values[candidate] for candidate in self.profile.strategies)
        updated_q = current_q + self.profile.learning_rate * (
            reward + self.profile.discount_factor * best_next_q - current_q
        )
        self.q_values[action] = updated_q
        self.action_counts[action] += 1

    def best_strategy(self) -> str:
        return max(self.profile.strategies, key=lambda action: self.q_values[action])


class AgenticMarketSimulation:
    def __init__(self, environment: MarketEnvironment, agent_profiles: List[AgentProfile]):
        self.environment = environment
        self.agents = {
            profile.name: StrategyRLAgent(profile=profile)
            for profile in agent_profiles
        }

    def run(self, cycles: int = 100) -> SimulationSummary:
        self.environment.reset()
        results: List[EpisodeResult] = []

        for cycle in range(1, cycles + 1):
            for agent_name, agent in self.agents.items():
                action = agent.choose_action()
                _, reward, kpi = self.environment.step(action)
                agent.learn(action=action, reward=reward)
                results.append(
                    EpisodeResult(
                        agent_name=agent_name,
                        cycle=cycle,
                        action=action,
                        reward=reward,
                        kpi=kpi,
                    )
                )

        return self._build_summary(results)

    async def run_async(self, cycles: int = 100) -> SimulationSummary:
        self.environment.reset()
        results: List[EpisodeResult] = []

        for cycle in range(1, cycles + 1):
            for agent_name, agent in self.agents.items():
                action = agent.choose_action()
                _, reward, kpi = self.environment.step(action)
                agent.learn(action=action, reward=reward)
                results.append(
                    EpisodeResult(
                        agent_name=agent_name,
                        cycle=cycle,
                        action=action,
                        reward=reward,
                        kpi=kpi,
                    )
                )
                await asyncio.sleep(0)

        return self._build_summary(results)

    def _build_summary(self, results: List[EpisodeResult]) -> SimulationSummary:
        rewards: Dict[str, List[float]] = defaultdict(list)
        action_rewards: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for item in results:
            rewards[item.agent_name].append(item.reward)
            action_rewards[item.agent_name][item.action].append(item.reward)

        best_strategy_by_agent = {}
        average_reward_by_agent = {}
        reward_improvement_by_agent = {}

        for agent_name, values in rewards.items():
            if not values:
                continue
            window = max(1, min(10, len(values) // 3))
            early_average = sum(values[:window]) / window
            late_average = sum(values[-window:]) / window

            average_reward_by_agent[agent_name] = sum(values) / len(values)
            reward_improvement_by_agent[agent_name] = late_average - early_average

            best_strategy_by_agent[agent_name] = max(
                action_rewards[agent_name].items(),
                key=lambda item: sum(item[1]) / max(len(item[1]), 1),
            )[0]

        return SimulationSummary(
            best_strategy_by_agent=best_strategy_by_agent,
            average_reward_by_agent=average_reward_by_agent,
            reward_improvement_by_agent=reward_improvement_by_agent,
            results=results,
        )


DEFAULT_STRATEGIES = ["aggressive_pricing", "value_targeting", "lead_engagement"]


def build_default_simulation(seed: int = 42) -> AgenticMarketSimulation:
    config = MarketConfig()
    environment = MarketEnvironment(config=config, seed=seed)
    profiles = [
        AgentProfile(name="consumer_growth_agent", strategies=DEFAULT_STRATEGIES),
        AgentProfile(name="competitor_response_agent", strategies=DEFAULT_STRATEGIES),
        AgentProfile(name="campaign_optimization_agent", strategies=DEFAULT_STRATEGIES),
    ]
    return AgenticMarketSimulation(environment=environment, agent_profiles=profiles)


def simulation_summary_to_dict(summary: SimulationSummary) -> dict:
    return {
        "best_strategy_by_agent": summary.best_strategy_by_agent,
        "average_reward_by_agent": summary.average_reward_by_agent,
        "reward_improvement_by_agent": summary.reward_improvement_by_agent,
        "results": [
            {
                "agent_name": episode.agent_name,
                "cycle": episode.cycle,
                "action": episode.action,
                "reward": episode.reward,
                "kpi": asdict(episode.kpi),
            }
            for episode in summary.results
        ],
    }
