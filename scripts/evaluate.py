import os
import sys
import json
import csv
import argparse
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.utils.logging_utils import setup_logger
from src.evaluation.retrieval_metrics import RetrievalEvaluator

logger = setup_logger("PAN11_Evaluator")

def evaluate_predictions(candidate_results_path: Path, ground_truth_path: Path) -> dict:
    evaluator = RetrievalEvaluator(k_values=[1, 5, 10, 20])

    # Load Ground Truth
    gt_map = {}
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                gt_map[rec["query_id"]] = rec["source_id"]

    query_evals = []
    with open(candidate_results_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                q_id = rec["query_id"]
                target_source = {gt_map[q_id]} if q_id in gt_map else set()
                ranked_docs = [c["source_document"] for c in rec.get("candidates", [])]
                
                m = evaluator.evaluate_query(ranked_docs, target_source)
                query_evals.append(m)

    summary = evaluator.evaluate_system(query_evals)
    return summary

def main():
    parser = argparse.ArgumentParser(description="Evaluate Candidate Fusion Results")
    parser.add_argument("--candidates", type=str, default="outputs/candidate_fusion_results.jsonl", help="Path to candidate_fusion_results.jsonl")
    parser.add_argument("--ground-truth", type=str, default="data/processed/ground_truth.jsonl", help="Path to ground_truth.jsonl")
    args = parser.parse_args()

    c_path = Path(args.candidates)
    gt_path = Path(args.ground_truth)

    if not c_path.exists():
        raise FileNotFoundError(f"Candidate results file not found at: {c_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found at: {gt_path}")

    summary = evaluate_predictions(c_path, gt_path)
    logger.info("\n" + "=" * 60)
    logger.info("CANDIDATE FUSION EVALUATION SUMMARY")
    logger.info("=" * 60)
    for k, v in summary.items():
        logger.info(f"{k.upper():<15}: {v:.4f}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
