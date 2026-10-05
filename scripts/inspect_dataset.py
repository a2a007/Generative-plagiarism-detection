import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.utils.logging_utils import setup_logger
from src.preprocessing.pan_parser import PANParser

logger = setup_logger("DatasetInspector")

def inspect_dataset(config: Dict[str, Any], parts_filter: List[str] = None) -> Dict[str, Any]:
    ext_root = Path(config["dataset"]["external_root"])
    source_root = ext_root / "source-document"
    susp_root = ext_root / "suspicious-document"

    logger.info(f"Inspecting dataset at external root: {ext_root}")
    if parts_filter:
        logger.info(f"Filtering suspicious documents to parts: {parts_filter}")

    # Helper to scan directory for files
    def scan_files(base_dir: Path, ext: str, filter_parts: List[str] = None) -> Dict[str, Path]:
        mapping = {}
        if not base_dir.exists():
            return mapping
        
        parts_to_scan = [base_dir / p for p in filter_parts] if filter_parts else list(base_dir.iterdir())
        for part_dir in parts_to_scan:
            if part_dir.is_dir():
                for f in part_dir.glob(f"*{ext}"):
                    filename = f.name
                    mapping[filename] = f
        return mapping

    logger.info("Scanning all source documents across corpus...")
    all_source_txt_files = scan_files(source_root, ".txt", filter_parts=None)
    all_source_xml_files = scan_files(source_root, ".xml", filter_parts=None)

    logger.info("Scanning suspicious documents...")
    susp_txt_files = scan_files(susp_root, ".txt", filter_parts=parts_filter)
    susp_xml_files = scan_files(susp_root, ".xml", filter_parts=parts_filter)

    # Check 1 - 4: File Counts
    num_all_source_txt = len(all_source_txt_files)
    num_all_source_xml = len(all_source_xml_files)
    num_susp_txt = len(susp_txt_files)
    num_susp_xml = len(susp_xml_files)

    logger.info(f"Total Source TXT across corpus: {num_all_source_txt}, XML: {num_all_source_xml}")
    logger.info(f"Inspected Suspicious TXT: {num_susp_txt}, XML: {num_susp_xml}")

    # Check 5: Missing TXT/XML pairs
    missing_source_xml_for_txt = [name for name in all_source_txt_files if name.replace(".txt", ".xml") not in all_source_xml_files]
    missing_source_txt_for_xml = [name for name in all_source_xml_files if name.replace(".xml", ".txt") not in all_source_txt_files]
    
    missing_susp_xml_for_txt = [name for name in susp_txt_files if name.replace(".txt", ".xml") not in susp_xml_files]
    missing_susp_txt_for_xml = [name for name in susp_xml_files if name.replace(".xml", ".txt") not in susp_txt_files]

    # Check 6, 7, 8, 9, 10, 11, 12: Parse XML annotations & validate
    parser = PANParser(str(ext_root))
    annotations, xml_errors, stats = parser.parse_directory(parts=parts_filter)

    xml_parse_failures = len(xml_errors)
    total_plagiarism_cases = len(annotations)

    suspicious_with_plagiarism: Set[str] = set()
    referenced_source_docs: Set[str] = set()

    for ann in annotations:
        suspicious_with_plagiarism.add(ann.suspicious_id)
        referenced_source_docs.add(ann.source_id)

    num_susp_with_plagiarism = len(suspicious_with_plagiarism)
    num_susp_without_plagiarism = num_susp_txt - num_susp_with_plagiarism
    num_unique_source_referenced = len(referenced_source_docs)

    # Check invalid source references (referenced source doc missing from disk globally)
    invalid_source_references = [src_id for src_id in referenced_source_docs if src_id not in all_source_txt_files]

    # Check character offset validity
    logger.info("Validating character offsets against raw document text lengths...")

    doc_char_lengths: Dict[str, int] = {}
    offset_errors = []

    def get_doc_length(doc_id: str, is_source: bool) -> int:
        if doc_id in doc_char_lengths:
            return doc_char_lengths[doc_id]
        
        file_map = all_source_txt_files if is_source else susp_txt_files
        if doc_id not in file_map:
            return -1
        
        file_path = file_map[doc_id]
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
                length = len(content)
                doc_char_lengths[doc_id] = length
                return length
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return -1

    this_offset_out_of_bounds = 0
    source_offset_out_of_bounds = 0

    for idx, ann in enumerate(annotations):
        # Validate suspicious offset
        susp_len = get_doc_length(ann.suspicious_id, is_source=False)
        if susp_len != -1:
            if ann.suspicious_offset + ann.suspicious_length > susp_len:
                this_offset_out_of_bounds += 1
                offset_errors.append({
                    "case_index": idx,
                    "type": "suspicious_offset_out_of_bounds",
                    "doc_id": ann.suspicious_id,
                    "offset": ann.suspicious_offset,
                    "length": ann.suspicious_length,
                    "doc_length": susp_len,
                    "xml_path": ann.xml_path
                })

        # Validate source offset
        src_len = get_doc_length(ann.source_id, is_source=True)
        if src_len != -1:
            if ann.source_offset + ann.source_length > src_len:
                source_offset_out_of_bounds += 1
                offset_errors.append({
                    "case_index": idx,
                    "type": "source_offset_out_of_bounds",
                    "doc_id": ann.source_id,
                    "offset": ann.source_offset,
                    "length": ann.source_length,
                    "doc_length": src_len,
                    "xml_path": ann.xml_path
                })

    report = {
        "dataset_root": str(ext_root),
        "parts_inspected": parts_filter if parts_filter else "ALL",
        "counts": {
            "total_corpus_source_txt_files": num_all_source_txt,
            "total_corpus_source_xml_files": num_all_source_xml,
            "suspicious_txt_files_inspected": num_susp_txt,
            "suspicious_xml_files_inspected": num_susp_xml,
            "total_plagiarism_cases": total_plagiarism_cases,
            "suspicious_docs_with_plagiarism": num_susp_with_plagiarism,
            "suspicious_docs_without_plagiarism": num_susp_without_plagiarism,
            "unique_source_docs_referenced": num_unique_source_referenced,
            "invalid_source_references_count": len(invalid_source_references)
        },
        "file_pairing_issues": {
            "missing_source_xml_for_txt_count": len(missing_source_xml_for_txt),
            "missing_source_txt_for_xml_count": len(missing_source_txt_for_xml),
            "missing_susp_xml_for_txt_count": len(missing_susp_xml_for_txt),
            "missing_susp_txt_for_xml_count": len(missing_susp_txt_for_xml)
        },
        "xml_parsing": {
            "xml_parse_failures": xml_parse_failures,
            "xml_error_details": xml_errors[:20]
        },
        "offset_validation": {
            "suspicious_offset_out_of_bounds_count": this_offset_out_of_bounds,
            "source_offset_out_of_bounds_count": source_offset_out_of_bounds,
            "offset_error_details": offset_errors[:20]
        },
        "invalid_source_references_list": invalid_source_references
    }

    return report

def format_text_report(report: Dict[str, Any]) -> str:
    lines = [
        "=" * 60,
        "PAN-11 PLAGIARISM CORPUS - DATASET VALIDATION REPORT",
        "=" * 60,
        f"Dataset Root: {report['dataset_root']}",
        f"Parts Inspected: {report['parts_inspected']}",
        "-" * 60,
        "1. FILE COUNTS:",
        f"   Total Corpus Source TXT Files : {report['counts']['total_corpus_source_txt_files']}",
        f"   Total Corpus Source XML Files : {report['counts']['total_corpus_source_xml_files']}",
        f"   Suspicious TXT Files Inspected: {report['counts']['suspicious_txt_files_inspected']}",
        f"   Suspicious XML Files Inspected: {report['counts']['suspicious_xml_files_inspected']}",
        "",
        "2. ANNOTATION STATS:",
        f"   Total Plagiarism Cases         : {report['counts']['total_plagiarism_cases']}",
        f"   Suspicious Docs WITH Plagiarism    : {report['counts']['suspicious_docs_with_plagiarism']}",
        f"   Suspicious Docs WITHOUT Plagiarism : {report['counts']['suspicious_docs_without_plagiarism']}",
        f"   Unique Source Documents Referenced : {report['counts']['unique_source_docs_referenced']}",
        f"   Invalid Source References          : {report['counts']['invalid_source_references_count']}",
        "",
        "3. PAIRING & PARSING:",
        f"   XML Parse Failures     : {report['xml_parsing']['xml_parse_failures']}",
        f"   Missing Source XMLs    : {report['file_pairing_issues']['missing_source_xml_for_txt_count']}",
        f"   Missing Suspicious XMLs: {report['file_pairing_issues']['missing_susp_xml_for_txt_count']}",
        "",
        "4. OFFSET VALIDATION:",
        f"   Suspicious Out-of-Bounds Offsets: {report['offset_validation']['suspicious_offset_out_of_bounds_count']}",
        f"   Source Out-of-Bounds Offsets    : {report['offset_validation']['source_offset_out_of_bounds_count']}",
        "=" * 60
    ]
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Inspect and validate PAN-11 Plagiarism Corpus")
    parser.add_argument("--part", type=str, help="Single part to inspect (e.g. 1 or part1)")
    parser.add_argument("--parts", nargs="+", type=str, help="Multiple parts to inspect (e.g. part1 part2 part3)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    parts_filter = []
    if args.part:
        p = f"part{args.part}" if not args.part.startswith("part") else args.part
        parts_filter.append(p)
    elif args.parts:
        for p in args.parts:
            parts_filter.append(f"part{p}" if not p.startswith("part") else p)

    report = inspect_dataset(config, parts_filter=parts_filter if parts_filter else None)

    reports_dir = Path("outputs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = Path(config["dataset"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    json_report_path = reports_dir / "dataset_report.json"
    txt_report_path = reports_dir / "dataset_report.txt"

    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    text_content = format_text_report(report)
    with open(txt_report_path, "w", encoding="utf-8") as f:
        f.write(text_content)

    logger.info("\n" + text_content)
    logger.info(f"Saved JSON report to: {json_report_path}")
    logger.info(f"Saved TXT report to: {txt_report_path}")

if __name__ == "__main__":
    main()
