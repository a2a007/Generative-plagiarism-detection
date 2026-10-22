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
        f"Input Document               : {report['input_document_name']}",
        f"Character Length             : {report['input_text_length_chars']:,} chars",
        f"Word Count                   : {report['input_text_word_count']:,} words",
        f"Total Evaluated Windows      : {report.get('total_evaluated_windows', report['total_passages_evaluated'])} windows",
        f"Confirmed Window Detections  : {report.get('confirmed_passage_detections', len(report['confirmed_detected_sources']))} detections",
        f"Merged Non-Overlapping Spans : {report.get('merged_input_regions_count', len(report['confirmed_detected_sources']))} regions",
        f"Unique Confirmed Sources     : {report.get('unique_source_documents_count', report['detected_sources_count'])} sources",
        f"Overall Verdict              : {report['overall_verdict']}",
        "-" * 70,
        f"CONFIRMED PLAGIARISM SOURCES ({report.get('unique_source_documents_count', report['detected_sources_count'])}):",
        "-" * 70
    ]

    if report["detected_sources_count"] == 0:
        lines.append("  No confirmed source document matches above threshold (0.50).")
    else:
        for idx, src in enumerate(report["confirmed_detected_sources"], start=1):
            eval_win = src.get("evaluated_windows_count", report["total_passages_evaluated"])
            conf_det = src.get("confirmed_passage_detections", src["detected_passages_count"])
            merged_reg = src.get("merged_input_regions_count", len(src["merged_input_offsets"]))
            align_ver = "Yes" if src.get("exact_alignment_verified", False) else "Fallback Alignment"

            lines.append(f"Source #{idx}: {src['source_document']}")
            lines.append(f"  - Decision                 : {src['source_level_decision']}")
            lines.append(f"  - Peak Confidence          : {src['highest_passage_confidence'] * 100:.2f}% (Tree Ensemble Vote)")
            lines.append(f"  - Evaluated Windows        : {eval_win} windows")
            lines.append(f"  - Confirmed Detections     : {conf_det} window detections")
            lines.append(f"  - Merged Input Regions     : {merged_reg} continuous region(s)")
            lines.append(f"  - Input Char Spans         : {src['merged_input_offsets']}")
            lines.append(f"  - Source Char Spans        : {src['merged_source_offsets']}")
            lines.append(f"  - Exact Alignment Verified : {align_ver}")
            lines.append(f"  - Best BM25 Score          : {src['best_bm25_score']}")
            lines.append(f"  - Dense Similarity         : {src['best_dense_similarity']}")
            lines.append(f"  - Best RRF Score           : {src['best_rrf_score']}")
            lines.append(f"  - Aligned Snippet:\n    \"{src['representative_snippet']}\"")
            lines.append("-" * 70)

    # Optional debug section showing comprehensive per-passage feature vectors & retrieval diagnostics
    if debug and "all_passage_evaluations" in report:
        lines.append("")
        lines.append("=" * 70)
        lines.append("DIAGNOSTIC LOGGING: PER-PASSAGE RETRIEVAL & CLASSIFICATION")
        lines.append("=" * 70)
        # Group by passage_id
        passages_dict: Dict[str, List[dict]] = {}
        for ev in report["all_passage_evaluations"]:
            pid = ev["passage_id"]
            if pid not in passages_dict:
                passages_dict[pid] = []
            passages_dict[pid].append(ev)

        for pid, ev_list in passages_dict.items():
            first_ev = ev_list[0]
            in_offsets = first_ev["input_passage_offsets"]
            w_count = first_ev["passage_word_count"]
            in_len = in_offsets[1] - in_offsets[0]
            snippet_preview = first_ev.get("input_passage_text", "").strip()[:100].replace("\n", " ")

            lines.append(f"Passage: {pid}")
            lines.append(f"  - Input Character Offsets : [{in_offsets[0]}, {in_offsets[1]}] ({in_len} chars)")
            lines.append(f"  - Word Count              : {w_count} words")
            lines.append(f"  - Passage Snippet         : \"{snippet_preview}...\"")
            lines.append(f"  - Retrieved Candidates ({len(ev_list)} evaluated):")

            for rank_idx, ev in enumerate(ev_list, start=1):
                db = ev.get("debug_features", {})
                b_rank = ev.get("bm25_rank", db.get("bm25_rank", "N/A"))
                d_rank = ev.get("dense_rank", db.get("dense_rank", "N/A"))
                f_rank = ev.get("fused_rank", db.get("fused_rank", rank_idx))
                evidence_status = "CONFIRMED" if db.get("has_textual_evidence") else "NONE"

                lines.append(f"    Candidate #{rank_idx}: {ev['source_document']}")
                lines.append(f"      * Decision        : {ev['plagiarism_decision']} (Probability: {ev['confidence_score']*100:.2f}%)")
                lines.append(f"      * Retrieval Ranks : Fused Rank: {f_rank} | BM25 Rank: {b_rank} | Dense Rank: {d_rank}")
                lines.append(f"      * Textual Evidence: {evidence_status}")

                if "feature_names" in db and "unscaled" in db and "scaled" in db:
                    lines.append("      * 10 Classifier Features (Raw vs Scaled):")
                    for i, (fname, raw_v, scl_v) in enumerate(zip(db["feature_names"], db["unscaled"], db["scaled"]), start=1):
                        lines.append(f"          {i:2d}. {fname:<24}: raw={raw_v:<10} | scaled={scl_v:<10}")
                lines.append("      " + "-" * 40)
            lines.append("-" * 70)

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
    parser.add_argument("--window-words", type=int, default=100, help="Words per passage window (default: 100)")
    parser.add_argument("--overlap-words", type=int, default=30, help="Overlap words between consecutive passages (default: 30)")
    parser.add_argument("--min-words", type=int, default=5, help="Minimum words required for an evaluated passage (default: 5)")
    parser.add_argument("--no-split-paragraphs", action="store_true", help="Disable paragraph-aware segmentation")
    parser.add_argument("--output-report", type=str, help="Save report to JSON file path")
    parser.add_argument("--debug-features", action="store_true", help="Print unscaled and scaled feature vectors and model probabilities")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose diagnostic logging (equivalent to --debug-features)")
    args = parser.parse_args()

    config = load_config(args.config)
    is_debug = args.debug_features or args.verbose

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
        overlap_words=args.overlap_words,
        min_words=args.min_words,
        split_paragraphs=not args.no_split_paragraphs
    )

    report = detector.detect_document(
        raw_text=raw_text,
        doc_name=doc_name,
        top_k=args.top_k,
        debug=is_debug
    )

    md_output = format_multi_source_markdown(report, debug=is_debug)
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
