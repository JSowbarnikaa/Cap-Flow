import unittest
from config import config

class TestConfigConsistency(unittest.TestCase):
    def test_config_completeness(self):
        self.assertTrue(len(config.SEEDS) >= 5)
        self.assertEqual(len(config.SIGNAL_STRENGTH_LEVELS), 5)
        self.assertIn("Tabular-Only", config.EXPERIMENTAL_CONDITIONS)
        self.assertGreater(len(config.TABULAR_FEATURE_SET), 0)
        self.assertGreater(len(config.SENTIMENT_FEATURE_SET), 0)
        self.assertEqual(
            len(config.FULL_FEATURE_SET),
            len(config.TABULAR_FEATURE_SET) + len(config.SENTIMENT_FEATURE_SET)
        )

if __name__ == "__main__":
    unittest.main()
