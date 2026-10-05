import os
import sys
import json
import pickle
import re
import argparse
from pathlib import Path
from typing import Dict, List, Any

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.utils.logging_utils import setup_logger
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.candidate_fusion import CandidateFusion
from src.classification.feature_extractor import FeatureExtractor, get_ngrams, jaccard_similarity

logger = setup_logger("PlagiarismDetector")

def run_plagiarism_detection(
    query_text: str,
    doc_name: str,
    config: dict,
    top_k: int = 10,
    confidence_threshold: float = 0.5
) -> Dict[str, Any]:
    indexes_dir = Path(config["dataset"]["indexes_dir"])
    bm25_index_path = indexes_dir / "bm25_index.pkl"
    pkl_path = Path("outputs/trained_plagiarism_classifier.pkl")

    if not bm25_index_path.exists():
        raise FileNotFoundError(f"BM25 index missing at {bm25_index_path}. Build index first.")
    if not pkl_path.exists():
        raise FileNotFoundError(f"Trained model pickle missing at {pkl_path}. Run train_and_evaluate_classifier.py first.")

    # 1. Load BM25 Retriever
    bm25_retriever = BM25Retriever(top_k=top_k*2)
    bm25_retriever.load_index(bm25_index_path)

    # 2. Load Dense Retriever
    dense_model_name = config.get("dense", {}).get("model", "sentence-transformers/all-MiniLM-L6-v2")
    dense_retriever = DenseRetriever(model_name=dense_model_name, top_k=top_k*2)
    dense_retriever.load_index(indexes_dir)

    # 3. Load Trained Classifier Pickle Model
    with open(pkl_path, "rb") as f:
        model_data = pickle.load(f)

    classifier = model_data["model"]
    scaler = model_data["scaler"]

    # 4. Execute Retrieval
    bm25_chunk_results = bm25_retriever.search(query_text, top_k=top_k*2)
    dense_chunk_results = dense_retriever.search(query_text, top_k=top_k*2)

    fusion_engine = CandidateFusion(rrf_k=config.get("fusion", {}).get("k", 60))
    fused_candidates = fusion_engine.fuse_results(bm25_chunk_results, dense_chunk_results, top_k=top_k*2)

    # 5. Extract features & predict with classifier model
    q_tokens = re.findall(r'\w+', query_text.lower())
    q_bigrams = get_ngrams(q_tokens, 2)
    q_trigrams = get_ngrams(q_tokens, 3)
    q_len = len(query_text)

    matched_sources = []
    has_plagiarism = False

    for cand in fused_candidates[:top_k]:
        src_doc = cand["source_document"]
        cand_text = cand.get("best_chunk_text", "")
        c_tokens = re.findall(r'\w+', cand_text.lower())
        c_bigrams = get_ngrams(c_tokens, 2)
        c_trigrams = get_ngrams(c_tokens, 3)

        word_jaccard = jaccard_similarity(set(q_tokens), set(c_tokens))
        bigram_jaccard = jaccard_similarity(q_bigrams, c_bigrams)
        trigram_jaccard = jaccard_similarity(q_trigrams, c_trigrams)

        c_len = len(cand_text)
        char_len_ratio = min(q_len, c_len) / max(q_len, c_len, 1)

        bm25_score = float(cand.get("bm25_score", 0.0))
        dense_score = float(cand.get("dense_score", 0.0))
        rrf_score = float(cand.get("rrf_score", 0.0))

        bm25_rank = cand.get("bm25_rank")
        bm25_rr = 1.0 / bm25_rank if bm25_rank else 0.0

        dense_rank = cand.get("dense_rank")
        dense_rr = 1.0 / dense_rank if dense_rank else 0.0

        fused_rank = cand.get("rank")
        fused_rr = 1.0 / fused_rank if fused_rank else 0.0

        feat = [[
            bm25_score,
            dense_score,
            rrf_score,
            bm25_rr,
            dense_rr,
            fused_rr,
            word_jaccard,
            bigram_jaccard,
            trigram_jaccard,
            char_len_ratio
        ]]

        feat_scaled = scaler.transform(feat)
        prob = float(classifier.predict_proba(feat_scaled)[0, 1]) if hasattr(classifier, "predict_proba") else float(classifier.predict(feat_scaled)[0])

        is_plag = prob >= confidence_threshold
        if is_plag:
            has_plagiarism = True

        matched_sources.append({
            "source_document": src_doc,
            "fused_rank": fused_rank,
            "confidence_score": round(prob, 4),
            "plagiarism_decision": "PLAGIARISM_MATCH" if is_plag else "NO_MATCH",
            "bm25_score": round(bm25_score, 4),
            "dense_similarity": round(dense_score, 4),
            "rrf_score": round(rrf_score, 4),
            "word_overlap_jaccard": round(word_jaccard, 4),
            "matched_chunk_id": cand.get("best_chunk_id"),
            "matched_source_snippet": cand_text[:300] + "..." if len(cand_text) > 300 else cand_text
        })

    report = {
        "input_document_name": doc_name,
        "input_text_length_chars": q_len,
        "input_text_word_count": len(q_tokens),
        "overall_verdict": "PLAGIARISM DETECTED" if has_plagiarism else "NO SIGNIFICANT PLAGIARISM DETECTED",
        "matched_sources_count": len(matched_sources),
        "top_source_matches": matched_sources
    }

    return report

def format_detection_markdown(report: dict) -> str:
    lines = [
        "=" * 70,
        "PLAGIARISM DETECTION ANALYSIS REPORT",
        "=" * 70,
        f"Input Document     : {report['input_document_name']}",
        f"Character Length   : {report['input_text_length_chars']} chars",
        f"Word Count         : {report['input_text_word_count']} words",
        f"Overall Verdict    : {report['overall_verdict']}",
        "-" * 70,
        "TOP MATCHED SOURCE DOCUMENTS:",
        "-" * 70
    ]

    for idx, match in enumerate(report["top_source_matches"], start=1):
        lines.append(f"Match #{idx}: {match['source_document']}")
        lines.append(f"  - Decision           : {match['plagiarism_decision']}")
        lines.append(f"  - Model Confidence   : {match['confidence_score'] * 100:.2f}%")
        lines.append(f"  - Fused Rank         : {match['fused_rank']}")
        lines.append(f"  - BM25 Score         : {match['bm25_score']}")
        lines.append(f"  - Dense Similarity   : {match['dense_similarity']}")
        lines.append(f"  - Word Overlap       : {match['word_overlap_jaccard']}")
        lines.append(f"  - Matched Source Text Snippet:\n    \"{match['matched_source_snippet']}\"")
        lines.append("-" * 70)

    lines.append("=" * 70)
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Run Individual Document Plagiarism Detection")
    parser.add_argument("--input", type=str, help="Path to input text file to check")
    parser.add_argument("--text", type=str, help="Raw text string to check")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top matching sources to report")
    parser.add_argument("--output-report", type=str, help="Save report to JSON file path")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found at: {input_path}")
        with open(input_path, "r", encoding="utf-8", errors="replace") as f:
            query_text = f.read()
        doc_name = input_path.name
    elif args.text:
        query_text = args.text
        doc_name = "Custom_User_Text_Input"
    else:
        # Default test using a sample suspicious document from part1
        sample_path = Path(config["dataset"]["external_root"]) / "suspicious-document" / "part1" / "suspicious-document00007.txt"
        if sample_path.exists():
            with open(sample_path, "r", encoding="utf-8") as f:
                query_text = f.read()[:2000]
            doc_name = "suspicious-document00007.txt (Sample)"
        else:
            raise ValueError("Please provide --input <filepath> or --text '<text>'")

    report = run_plagiarism_detection(query_text, doc_name, config, top_k=args.top_k)
    md_output = format_detection_markdown(report)

    logger.info("\n" + md_output)

    if args.output_report:
        out_path = Path(args.output_report)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved detection report to: {out_path}")

if __name__ == "__main__":
    main()
