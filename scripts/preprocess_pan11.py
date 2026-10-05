import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.utils.logging_utils import setup_logger
from src.preprocessing.pan_parser import PANParser
from src.preprocessing.document_loader import DocumentLoader
from src.preprocessing.ground_truth import GroundTruthExtractor
from src.preprocessing.chunker import DocumentChunker

logger = setup_logger("PAN11_Preprocessor")

def preprocess_pan11(config: Dict[str, Any], parts_filter: List[str] = None):
    ext_root = Path(config["dataset"]["external_root"])
    processed_dir = Path(config["dataset"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting PAN-11 Preprocessing. External Root: {ext_root}")
    if parts_filter:
        logger.info(f"Filter parts: {parts_filter}")

    # 1. Parse XML Annotations
    logger.info("Parsing suspicious document XML annotations...")
    parser = PANParser(str(ext_root))
    annotations, xml_errors, stats = parser.parse_directory(parts=parts_filter)

    logger.info(f"Parsed {len(annotations)} plagiarism annotations with {len(xml_errors)} errors.")

    # 2. Extract Ground Truth & Queries
    loader = DocumentLoader(str(ext_root))
    gt_extractor = GroundTruthExtractor(loader)
    
    logger.info("Building ground truth records and queries...")
    gt_records, query_records = gt_extractor.build_ground_truth_and_queries(annotations)

    gt_path = processed_dir / "ground_truth.jsonl"
    query_path = processed_dir / "queries.jsonl"
    gt_extractor.save_jsonl(gt_records, gt_path)
    gt_extractor.save_jsonl(query_records, query_path)
    logger.info(f"Saved {len(gt_records)} ground truth records to: {gt_path}")
    logger.info(f"Saved {len(query_records)} query records to: {query_path}")

    # 3. Load & Save Source and Suspicious Documents
    source_docs_path = processed_dir / "source_documents.jsonl"
    susp_docs_path = processed_dir / "suspicious_documents.jsonl"
    chunks_path = processed_dir / "source_chunks.jsonl"

    words_per_chunk = config.get("chunking", {}).get("words_per_chunk", 512)
    overlap_words = config.get("chunking", {}).get("overlap_words", 100)
    chunker = DocumentChunker(words_per_chunk=words_per_chunk, overlap_words=overlap_words)

    # Gather source documents referenced in ground truth or for the selected parts
    logger.info("Loading source documents for chunking...")
    # For a dev/test run on part1, we load source documents from part1 or all referenced source docs
    referenced_source_ids = {gt["source_id"] for gt in gt_records}
    
    # Load all source files in parts_filter if specified, plus any referenced source files
    source_files_to_load = set()
    if parts_filter:
        for part in parts_filter:
            p_dir = ext_root / "source-document" / (f"part{part}" if not part.startswith("part") else part)
            if p_dir.exists():
                for f in p_dir.glob("*.txt"):
                    source_files_to_load.add(f.name)
    else:
        for f in (ext_root / "source-document").rglob("*.txt"):
            source_files_to_load.add(f.name)
            
    # Always ensure referenced source docs are included
    source_files_to_load.update(referenced_source_ids)

    logger.info(f"Total source documents to load & chunk: {len(source_files_to_load)}")

    total_chunks = 0
    with open(source_docs_path, "w", encoding="utf-8") as f_src, \
         open(chunks_path, "w", encoding="utf-8") as f_chk:
        
        for doc_id in sorted(list(source_files_to_load)):
            path = loader.get_source_file_path(doc_id)
            if not path or not path.exists():
                logger.warning(f"Source document {doc_id} not found on disk.")
                continue
            
            text = loader.load_document_raw(path)
            f_src.write(json.dumps({"doc_id": doc_id, "text": text, "language": "en"}, ensure_ascii=False) + "\n")

            # Chunking
            doc_chunks = chunker.chunk_document(doc_id, text)
            for chk in doc_chunks:
                f_chk.write(json.dumps(chk, ensure_ascii=False) + "\n")
                total_chunks += 1

    logger.info(f"Saved source documents to: {source_docs_path}")
    logger.info(f"Saved {total_chunks} source chunks to: {chunks_path}")

    # Load suspicious documents
    susp_doc_ids = {q["suspicious_id"] for q in query_records}
    if parts_filter:
        for part in parts_filter:
            p_dir = ext_root / "suspicious-document" / (f"part{part}" if not part.startswith("part") else part)
            if p_dir.exists():
                for f in p_dir.glob("*.txt"):
                    susp_doc_ids.add(f.name)

    logger.info(f"Total suspicious documents to save: {len(susp_doc_ids)}")
    with open(susp_docs_path, "w", encoding="utf-8") as f_susp:
        for doc_id in sorted(list(susp_doc_ids)):
            path = loader.get_suspicious_file_path(doc_id)
            if not path or not path.exists():
                continue
            text = loader.load_document_raw(path)
            f_susp.write(json.dumps({"doc_id": doc_id, "text": text, "language": "en"}, ensure_ascii=False) + "\n")

    logger.info(f"Saved suspicious documents to: {susp_docs_path}")
    logger.info("PAN-11 Preprocessing Completed Successfully!")

def main():
    parser = argparse.ArgumentParser(description="Preprocess PAN-11 Plagiarism Corpus")
    parser.add_argument("--part", type=str, help="Single part to preprocess (e.g. 1 or part1)")
    parser.add_argument("--parts", nargs="+", type=str, help="Multiple parts to preprocess (e.g. part1 part2)")
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

    preprocess_pan11(config, parts_filter=parts_filter if parts_filter else None)

if __name__ == "__main__":
    main()
