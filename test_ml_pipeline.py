import json
import unittest
from pathlib import Path

import joblib
import pandas as pd

FEATURES = ["attendance", "internal_marks", "assignment_marks", "previous_score"]


class TestMLPipeline(unittest.TestCase):

    def test_artifacts_exist(self):
        for name in ("student_result_model.pkl", "metrics.json", "student_results.csv"):
            self.assertTrue(Path(name).exists(), f"{name} is missing")

    def test_metrics_accuracy_valid(self):
        accuracy = json.loads(Path("metrics.json").read_text())["accuracy"]
        self.assertGreaterEqual(accuracy, 0.8)
        self.assertLessEqual(accuracy, 1.0)

    def test_model_predicts_pass_and_fail(self):
        model = joblib.load("student_result_model.pkl")
        high = pd.DataFrame([[90, 85, 88, 80]], columns=FEATURES)
        low = pd.DataFrame([[55, 30, 40, 35]], columns=FEATURES)
        self.assertEqual(int(model.predict(high)[0]), 1)
        self.assertEqual(int(model.predict(low)[0]), 0)


if __name__ == "__main__":
    unittest.main()
