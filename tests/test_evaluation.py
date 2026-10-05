import unittest
from src.evaluation.retrieval_metrics import RetrievalEvaluator

class TestRetrievalEvaluator(unittest.TestCase):
    def test_metrics_calculation(self):
        evaluator = RetrievalEvaluator(k_values=[1, 5, 10, 20])

        ranked_docs = [
            "source-document05726.txt",
            "source-document07450.txt",
            "source-document00875.txt"
        ]
        target_source = {"source-document07450.txt"}

        metrics = evaluator.evaluate_query(ranked_docs, target_source)

        self.assertEqual(metrics["recall@1"], 0.0)
        self.assertEqual(metrics["recall@5"], 1.0)
        self.assertEqual(metrics["recall@10"], 1.0)
        self.assertEqual(metrics["mrr"], 0.5)  # rank 2 -> 1/2

if __name__ == "__main__":
    unittest.main()
