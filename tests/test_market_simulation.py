import unittest

from agent.market_simulation import build_default_simulation
from agent.stock_graph import run_market_simulation_graph


class TestMarketSimulation(unittest.TestCase):
    def test_simulation_generates_kpis_and_results(self):
        simulation = build_default_simulation(seed=10)
        summary = simulation.run(cycles=40)

        self.assertTrue(summary.results)
        self.assertTrue(summary.best_strategy_by_agent)
        self.assertTrue(summary.average_reward_by_agent)

    def test_learning_loop_improves_rewards_for_majority_agents(self):
        simulation = build_default_simulation(seed=21)
        summary = simulation.run(cycles=120)

        improvements = summary.reward_improvement_by_agent
        positive_count = sum(1 for value in improvements.values() if value > 0)

        self.assertGreaterEqual(positive_count, 2)

    def test_langgraph_flow_returns_strategy_summary(self):
        result = run_market_simulation_graph(cycles=60)

        self.assertIn("best_strategy_by_agent", result)
        self.assertIn("reward_improvement_by_agent", result)
        self.assertEqual(len(result["best_strategy_by_agent"]), 3)


if __name__ == "__main__":
    unittest.main()
