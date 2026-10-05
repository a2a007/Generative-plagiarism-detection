import numpy as np
from typing import List, Dict, Any, Set

class RetrievalEvaluator:
    """
    Evaluator for document-level plagiarism retrieval.
    Computes Recall@1, Recall@5, Recall@10, Recall@20, and MRR.
    """
    def __init__(self, k_values: List[int] = [1, 5, 10, 20]):
        self.k_values = sorted(k_values)

    def evaluate_query(self, ranked_source_docs: List[str], ground_truth_sources: Set[str]) -> Dict[str, float]:
        """
        Evaluates retrieval performance for a single query against target ground truth source(s).
        """
        metrics = {}
        if not ground_truth_sources:
            for k in self.k_values:
                metrics[f"recall@{k}"] = 0.0
            metrics["mrr"] = 0.0
            return metrics

        # Compute Recall@K
        for k in self.k_values:
            top_k_docs = set(ranked_source_docs[:k])
            hits = top_k_docs.intersection(ground_truth_sources)
            # Binary hit (is target source retrieved in top k?)
            metrics[f"recall@{k}"] = 1.0 if len(hits) > 0 else 0.0

        # Compute Reciprocal Rank (MRR)
        rr = 0.0
        for rank, doc in enumerate(ranked_source_docs, start=1):
            if doc in ground_truth_sources:
                rr = 1.0 / rank
                break
        metrics["mrr"] = rr

        return metrics

    def evaluate_system(self, query_eval_results: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Averages metrics across all queries.
        """
        if not query_eval_results:
            return {f"recall@{k}": 0.0 for k in self.k_values} | {"mrr": 0.0}

        aggregated = {}
        keys = query_eval_results[0].keys()
        for key in keys:
            vals = [res[key] for res in query_eval_results]
            aggregated[key] = float(np.mean(vals))

        return aggregated
