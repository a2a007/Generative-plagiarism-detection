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

def format_compact_terminal_report(report: dict) -> str:
    """
    Formats detection results into a clean, compact terminal report.
    Displays every confirmed matched passage grouped by source document without overflowing.
    """
    confirmed = report.get("confirmed_detected_sources", [])

    # Sort confirmed sources by their first appearance in the input document
    sorted_sources = sorted(
        confirmed,
        key=lambda s: min(h["input_passage_offsets"][0] for h in s.get("matching_passage_details", []))
        if s.get("matching_passage_details") else 0
    )

    total_sources = len(sorted_sources)
    total_passages = sum(len(src.get("matching_passage_details", [])) for src in sorted_sources)

    verdict = "PLAGIARISM DETECTED" if total_passages > 0 else "NO PLAGIARISM DETECTED"

    lines = [
        "=" * 60,
        "PLAGIARISM DETECTION REPORT",
        "=" * 60,
        f"Input File       : {report.get('input_document_name', 'Unknown')}",
        f"Overall Verdict  : {verdict}",
        f"Sources Detected : {total_sources}",
        f"Passages Detected: {total_passages}",
        "=" * 60
    ]

    if total_passages == 0:
        return "\n".join(lines)

    for idx, src in enumerate(sorted_sources, start=1):
        src_name = src.get("source_document", f"Source {idx}")
        hits = src.get("matching_passage_details", [])

        # Sort hits in document order by input passage start offset
        sorted_hits = sorted(hits, key=lambda h: h.get("input_passage_offsets", [0])[0])

        lines.append("")
        lines.append(f"SOURCE {idx}: {src_name}")
        lines.append("-" * 60)
        lines.append(f"Passages Detected: {len(sorted_hits)}")
        lines.append("")

        for p_idx, ev in enumerate(sorted_hits, start=1):
            pid = ev.get("passage_id", f"passage_{p_idx:04d}")
            conf = ev.get("confidence_score", 0.0)
            conf_str = f"{round(conf * 100)}%"

            in_offsets = ev.get("input_passage_offsets", [0, 0])
            in_loc = f"Characters {in_offsets[0]}-{in_offsets[1]}"

            src_offsets = ev.get("source_passage_offsets")
            if (
                src_offsets
                and len(src_offsets) == 2
                and src_offsets[0] is not None
                and src_offsets[1] is not None
                and (src_offsets[0] > 0 or src_offsets[1] > 0)
            ):
                src_loc = f"Characters {src_offsets[0]}-{src_offsets[1]}"
            else:
                src_loc = "Not available"

            raw_text = ev.get("input_passage_text", "").strip()
            if not raw_text:
                raw_text = ev.get("matched_source_snippet", "").strip()
            raw_text = re.sub(r'\s+', ' ', raw_text)
            if len(raw_text) > 150:
                preview = raw_text[:150].strip() + "..."
            else:
                preview = raw_text

            p_prefix = f"[{p_idx}] Passage ID"
            lines.append(f"{p_prefix:<20} : {pid}")
            lines.append(f"{'    Confidence':<20} : {conf_str}")
            lines.append(f"{'    Input Location':<20} : {in_loc}")
            lines.append(f"{'    Matched Source':<20} : {src_name}")
            lines.append(f"{'    Source Location':<20} : {src_loc}")
            lines.append(f"{'    Matched Text':<20} : {preview}")
            lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


# Alias for backward compatibility
format_multi_source_markdown = format_compact_terminal_report


def format_verbose_diagnostics(report: dict) -> str:
    """Formats optional deep diagnostic logs when --verbose or --debug-features is specified."""
    lines = [
        "",
        "=" * 60,
        "VERBOSE DIAGNOSTICS: CANDIDATE FEATURE VECTORS & PROBABILITIES",
        "=" * 60
    ]
    for ev in report.get("all_passage_evaluations", []):
        if "debug_features" in ev:
            db = ev["debug_features"]
            lines.append(f"Passage: {ev['passage_id']} -> Candidate: {ev['source_document']}")
            lines.append(f"  - Decision: {ev['plagiarism_decision']} (Probability: {ev['confidence_score']*100:.2f}%)")
            lines.append(f"  - Model: {db.get('model_type')} | Classes: {db.get('classes')}")
            lines.append("  - 10 Classifier Features:")
            for fname, raw_v, scl_v in zip(db.get("feature_names", []), db.get("unscaled", []), db.get("scaled", [])):
                lines.append(f"      {fname:<24}: raw={raw_v:<10} | scaled={scl_v:<10}")
            lines.append("-" * 40)
    lines.append("=" * 60)
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
    parser.add_argument("--debug-features", action="store_true", help="Include feature vector debugging in report and display verbose diagnostics")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose diagnostic logging")
    args = parser.parse_args()

    config = load_config(args.config)
    is_debug = args.debug_features or args.verbose

    # Silence background info logs unless verbose mode is explicitly enabled
    if not is_debug:
        import logging as py_logging
        for log_name in ["PAN11_Pipeline", "PlagiarismDetector", "PassagePlagiarismDetector", "BM25Retriever", "DenseRetriever", "CandidateFusion"]:
            py_logging.getLogger(log_name).setLevel(py_logging.WARNING)
        try:
            from transformers.utils import logging as tf_logging
            tf_logging.set_verbosity_error()
        except Exception:
            pass
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

    # Print clean compact terminal report
    compact_output = format_compact_terminal_report(report)
    print("\n" + compact_output)

    if is_debug:
        verbose_output = format_verbose_diagnostics(report)
        print("\n" + verbose_output)

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
        if is_debug:
            logger.info(f"Saved detection report to: {out_path}")

if __name__ == "__main__":
    main()
