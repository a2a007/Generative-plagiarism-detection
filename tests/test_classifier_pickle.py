import unittest
import pickle
from pathlib import Path

class TestClassifierPickle(unittest.TestCase):
    def test_load_pickle_model(self):
        pkl_path = Path("outputs/trained_plagiarism_classifier.pkl")
        self.assertTrue(pkl_path.exists(), "Pickle file outputs/trained_plagiarism_classifier.pkl must exist.")

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        self.assertIn("model", data)
        self.assertIn("scaler", data)
        self.assertIn("feature_names", data)
        self.assertIn("metrics", data)

        model = data["model"]
        scaler = data["scaler"]
        metrics = data["metrics"]

        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertGreaterEqual(metrics["f1_binary"], 0.0)

if __name__ == "__main__":
    unittest.main()
