import os
import sys
import json
import pickle
import re
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.utils.logging_utils import setup_logger
from src.classification.passage_detector import PassagePlagiarismDetector, extract_best_matching_snippet

logger = setup_logger("PlagiarismDetector")

def format_multi_source_markdown(report: dict, debug: bool = False) -> str:
    lines = [
        "=" * 70,
        "PASSAGE-LEVEL MULTI-SOURCE PLAGIARISM DETECTION ANALYSIS REPORT",
        "=" * 70,
        f"Input Document     : {report['input_document_name']}",
        f"Character Length   : {report['input_text_length_chars']:,} chars",
        f"Word Count         : {report['input_text_word_count']:,} words",
        f"Passages Evaluated : {report['total_passages_evaluated']}",
        f"Overall Verdict    : {report['overall_verdict']}",
        "-" * 70,
        f"CONFIRMED PLAGIARISM SOURCES ({report['detected_sources_count']}):",
        "-" * 70
    ]

    if report["detected_sources_count"] == 0:
        lines.append("  No confirmed source document matches above threshold (0.50).")
    else:
        for idx, src in enumerate(report["confirmed_detected_sources"], start=1):
            lines.append(f"Source #{idx}: {src['source_document']}")
            lines.append(f"  - Decision           : {src['source_level_decision']}")
            lines.append(f"  - Peak Confidence    : {src['highest_passage_confidence'] * 100:.2f}%")
            lines.append(f"  - Detected Passages  : {src['detected_passages_count']}")
            lines.append(f"  - Input Char Spans   : {src['merged_input_offsets']}")
            lines.append(f"  - Source Char Spans  : {src['merged_source_offsets']}")
            lines.append(f"  - Best BM25 Score    : {src['best_bm25_score']}")
            lines.append(f"  - Dense Similarity   : {src['best_dense_similarity']}")
            lines.append(f"  - Best RRF Score     : {src['best_rrf_score']}")
            lines.append(f"  - Aligned Snippet:\n    \"{src['representative_snippet']}\"")
            lines.append("-" * 70)

    # Optional debug section showing feature vectors
    if debug and "all_passage_evaluations" in report:
        lines.append("")
        lines.append("=" * 70)
        lines.append("DEBUG: CANDIDATE FEATURE VECTORS & PROBABILITIES")
        lines.append("=" * 70)
        for ev in report["all_passage_evaluations"]:
            if "debug_features" in ev:
                db = ev["debug_features"]
                lines.append(f"Passage: {ev['passage_id']} -> Candidate: {ev['source_document']}")
                lines.append(f"  - Decision: {ev['plagiarism_decision']} (Prob: {ev['confidence_score']*100:.2f}%)")
                lines.append(f"  - Model Type: {db['model_type']} | Classes: {db['classes']}")
                lines.append("  - Feature Values:")
                for fname, raw_v, scl_v in zip(db["feature_names"], db["unscaled"], db["scaled"]):
                    lines.append(f"      {fname:<24}: raw={raw_v:<10} | scaled={scl_v:<10}")
                lines.append("-" * 50)

    # Top non-matching candidate sources retrieved
    non_matches = report.get("retrieved_non_matching_sources", [])
    if non_matches:
        lines.append("")
        lines.append(f"TOP RETRIEVED NON-MATCHING SOURCES ({len(non_matches[:3])} of {len(non_matches)} shown):")
        lines.append("-" * 70)
        for idx, nm in enumerate(non_matches[:3], start=1):
            lines.append(f"Candidate #{idx}: {nm['source_document']}")
            lines.append(f"  - Decision           : {nm['source_level_decision']}")
            lines.append(f"  - Peak Confidence    : {nm['highest_passage_confidence'] * 100:.2f}%")
            lines.append(f"  - Best BM25 Score    : {nm['best_bm25_score']}")
            lines.append(f"  - Dense Similarity   : {nm['best_dense_similarity']}")
            lines.append(f"  - Best RRF Score     : {nm['best_rrf_score']}")
            lines.append("-" * 70)

    lines.append("=" * 70)
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Run Passage-Level Multi-Source Plagiarism Detection")
    parser.add_argument("--input", "--file", type=str, dest="input", help="Path to input text file to check")
    parser.add_argument("--text", type=str, help="Raw text string to check")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top candidate sources per passage")
    parser.add_argument("--confidence-threshold", type=float, default=0.50, help="Classification confidence threshold")
    parser.add_argument("--window-words", type=int, default=150, help="Words per passage window")
    parser.add_argument("--overlap-words", type=int, default=50, help="Overlap words between consecutive passages")
    parser.add_argument("--output-report", type=str, help="Save report to JSON file path")
    parser.add_argument("--debug-features", action="store_true", help="Print unscaled and scaled feature vectors and model probabilities")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found at: {input_path}")
        with open(input_path, "r", encoding="utf-8", errors="replace") as f:
            raw_text = f.read()
        doc_name = input_path.name
    elif args.text:
        raw_text = args.text
        doc_name = "Custom_User_Text_Input"
    else:
        sample_path = Path(config["dataset"]["external_root"]) / "suspicious-document" / "part1" / "suspicious-document00007.txt"
        if sample_path.exists():
            with open(sample_path, "r", encoding="utf-8") as f:
                raw_text = f.read()[:2000]
            doc_name = "suspicious-document00007.txt (Sample)"
        else:
            raise ValueError("Please provide --input <filepath> or --text '<text>'")

    detector = PassagePlagiarismDetector(
        config=config,
        confidence_threshold=args.confidence_threshold,
        window_words=args.window_words,
        overlap_words=args.overlap_words
    )

    report = detector.detect_document(
        raw_text=raw_text,
        doc_name=doc_name,
        top_k=args.top_k,
        debug=args.debug_features
    )

    md_output = format_multi_source_markdown(report, debug=args.debug_features)
    logger.info("\n" + md_output)

    if args.output_report:
        out_path = Path(args.output_report)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        def json_encoder(o):
            if isinstance(o, (np.integer, np.int32, np.int64)):
                return int(o)
            if isinstance(o, (np.floating, np.float32, np.float64)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return str(o)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=json_encoder)
        logger.info(f"Saved detection report to: {out_path}")

if __name__ == "__main__":
    main()
