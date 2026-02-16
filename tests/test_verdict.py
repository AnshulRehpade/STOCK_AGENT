import unittest
from agent.verdict import format_verdict

class TestVerdict(unittest.TestCase):
    def test_format(self):
        result = format_verdict("Rise", "Strong momentum.", "High", "Optimistic")
        self.assertIn("📈 Final Verdict: Rise", result)
        self.assertIn("💬 Reasoning: Strong momentum.", result)
        self.assertIn("📊 Confidence Level: High", result)
        self.assertIn("💰 Investor Outlook: Optimistic", result)

if __name__ == "__main__":
    unittest.main()