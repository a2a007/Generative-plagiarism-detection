import os
import sys
import json
import csv
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.utils.logging_utils import setup_logger
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.candidate_fusion import CandidateFusion
from src.evaluation.retrieval_metrics import RetrievalEvaluator

logger = setup_logger("PAN11_Experiment")

def run_experiment(config: Dict[str, Any], max_queries: int = None):
    processed_dir = Path(config["dataset"]["processed_dir"])
    indexes_dir = Path(config["dataset"]["indexes_dir"])
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    gt_path = processed_dir / "ground_truth.jsonl"
    query_path = processed_dir / "queries.jsonl"
    bm25_index_path = indexes_dir / "bm25_index.pkl"

    if not gt_path.exists() or not query_path.exists():
        raise FileNotFoundError("Processed ground_truth.jsonl or queries.jsonl not found. Run preprocess_pan11.py first.")

    # Load Ground Truth
    logger.info(f"Loading ground truth from: {gt_path}")
    gt_map: Dict[str, Dict[str, Any]] = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                gt_map[rec["query_id"]] = rec

    # Load Queries
    logger.info(f"Loading queries from: {query_path}")
    queries: List[Dict[str, Any]] = []
    with open(query_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                queries.append(rec)
                if max_queries and len(queries) >= max_queries:
                    break

    logger.info(f"Loaded {len(queries)} queries for experiment.")

    # Load Retrievers
    top_k = config.get("bm25", {}).get("top_k", 100)
    
    logger.info(f"Loading BM25 Index from: {bm25_index_path}")
    bm25_retriever = BM25Retriever(top_k=top_k)
    bm25_retriever.load_index(bm25_index_path)

    dense_model_name = config.get("dense", {}).get("model", "sentence-transformers/all-MiniLM-L6-v2")
    logger.info(f"Loading Dense FAISS Index from: {indexes_dir}")
    dense_retriever = DenseRetriever(model_name=dense_model_name, top_k=top_k)
    dense_retriever.load_index(indexes_dir)

    rrf_k = config.get("fusion", {}).get("k", 60)
    fusion_engine = CandidateFusion(rrf_k=rrf_k)
    evaluator = RetrievalEvaluator(k_values=[1, 5, 10, 20])

    bm25_query_evals = []
    dense_query_evals = []
    fusion_query_evals = []

    candidate_results_path = outputs_dir / "candidate_fusion_results.jsonl"
    logger.info("Executing Retrieval Experiment across queries...")

    with open(candidate_results_path, "w", encoding="utf-8") as f_out:
        for q in tqdm(queries, desc="Evaluating Queries"):
            q_id = q["query_id"]
            q_text = q["query_text"]

            if not q_text.strip():
                continue

            gt_rec = gt_map.get(q_id)
            if not gt_rec:
                continue

            target_source_doc = {gt_rec["source_id"]}

            # 1. BM25 Search
            bm25_chunk_results = bm25_retriever.search(q_text, top_k=top_k)
            bm25_doc_candidates = fusion_engine.aggregate_chunk_results_to_doc(bm25_chunk_results)
            bm25_ranked_docs = [d["source_document"] for d in bm25_doc_candidates]
            bm25_eval = evaluator.evaluate_query(bm25_ranked_docs, target_source_doc)
            bm25_query_evals.append(bm25_eval)

            # 2. Dense Search
            dense_chunk_results = dense_retriever.search(q_text, top_k=top_k)
            dense_doc_candidates = fusion_engine.aggregate_chunk_results_to_doc(dense_chunk_results)
            dense_ranked_docs = [d["source_document"] for d in dense_doc_candidates]
            dense_eval = evaluator.evaluate_query(dense_ranked_docs, target_source_doc)
            dense_query_evals.append(dense_eval)

            # 3. Candidate Fusion
            fused_doc_candidates = fusion_engine.fuse_results(bm25_chunk_results, dense_chunk_results, top_k=top_k)
            fused_ranked_docs = [d["source_document"] for d in fused_doc_candidates]
            fusion_eval = evaluator.evaluate_query(fused_ranked_docs, target_source_doc)
            fusion_query_evals.append(fusion_eval)

            # Save candidate record for future cross-encoder stage
            fusion_record = {
                "query_id": q_id,
                "suspicious_document": q["suspicious_id"],
                "query_offset": q["suspicious_offset"],
                "query_length": q["suspicious_length"],
                "query_text": q_text,
                "ground_truth_source": gt_rec["source_id"],
                "candidates": fused_doc_candidates[:20]
            }
            f_out.write(json.dumps(fusion_record, ensure_ascii=False) + "\n")

    # Aggregate Systems Evaluation
    bm25_summary = evaluator.evaluate_system(bm25_query_evals)
    dense_summary = evaluator.evaluate_system(dense_query_evals)
    fusion_summary = evaluator.evaluate_system(fusion_query_evals)

    logger.info("\n" + "=" * 70)
    logger.info("HYBRID LEXICAL-SEMANTIC PLAGIARISM RETRIEVAL EXPERIMENT RESULTS")
    logger.info("=" * 70)
    header = f"{'System':<25} | {'Recall@1':<10} | {'Recall@5':<10} | {'Recall@10':<10} | {'Recall@20':<10} | {'MRR':<10}"
    logger.info(header)
    logger.info("-" * 70)

    def print_metrics(name, m):
        logger.info(f"{name:<25} | {m['recall@1']:<10.4f} | {m['recall@5']:<10.4f} | {m['recall@10']:<10.4f} | {m['recall@20']:<10.4f} | {m['mrr']:<10.4f}")

    print_metrics("BM25 (Lexical)", bm25_summary)
    print_metrics("Dense (Semantic)", dense_summary)
    print_metrics("BM25 + Dense Fusion", fusion_summary)
    logger.info("=" * 70)

    # Save results.csv
    csv_path = outputs_dir / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["System", "Recall@1", "Recall@5", "Recall@10", "Recall@20", "MRR"])
        for sys_name, s_dict in [("BM25", bm25_summary), ("Dense", dense_summary), ("Fusion", fusion_summary)]:
            writer.writerow([sys_name, s_dict["recall@1"], s_dict["recall@5"], s_dict["recall@10"], s_dict["recall@20"], s_dict["mrr"]])

    logger.info(f"Saved results summary CSV to: {csv_path}")

    # Save experiment_config.json
    exp_config_path = outputs_dir / "experiment_config.json"
    exp_config = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": config,
        "queries_count": len(bm25_query_evals),
        "results": {
            "bm25": bm25_summary,
            "dense": dense_summary,
            "fusion": fusion_summary
        }
    }
    with open(exp_config_path, "w", encoding="utf-8") as f_exp:
        json.dump(exp_config, f_exp, indent=2)

    logger.info(f"Saved experiment configuration & metadata to: {exp_config_path}")

def main():
    parser = argparse.ArgumentParser(description="Run Hybrid Lexical-Semantic Plagiarism Detection Experiment")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--max-queries", type=int, help="Limit maximum queries for rapid experiment testing")
    args = parser.parse_args()

    config = load_config(args.config)
    run_experiment(config, max_queries=args.max_queries)

if __name__ == "__main__":
    main()
