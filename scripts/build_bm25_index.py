import os
import sys
import json
import argparse
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.utils.logging_utils import setup_logger
from src.retrieval.bm25_retriever import BM25Retriever

logger = setup_logger("BM25_IndexBuilder")

def build_bm25_index(config: dict, force: bool = False):
    processed_dir = Path(config["dataset"]["processed_dir"])
    indexes_dir = Path(config["dataset"]["indexes_dir"])
    indexes_dir.mkdir(parents=True, exist_ok=True)

    index_path = indexes_dir / "bm25_index.pkl"
    chunks_path = processed_dir / "source_chunks.jsonl"

    if index_path.exists() and not force:
        logger.info(f"BM25 index already exists at {index_path}. Use --force to rebuild.")
        return

    if not chunks_path.exists():
        raise FileNotFoundError(f"Source chunks file not found at: {chunks_path}. Run preprocess_pan11.py first.")

    logger.info(f"Loading source chunks from: {chunks_path}")
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    logger.info(f"Loaded {len(chunks)} chunks. Building BM25 index...")
    top_k = config.get("bm25", {}).get("top_k", 100)
    retriever = BM25Retriever(top_k=top_k)
    retriever.build_index(chunks)

    logger.info(f"Saving BM25 index to: {index_path}")
    retriever.save_index(index_path)
    logger.info("BM25 Index Built Successfully!")

def main():
    parser = argparse.ArgumentParser(description="Build BM25 Index for PAN-11 Source Chunks")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--force", action="store_true", help="Force rebuild index even if exists")
    args = parser.parse_args()

    config = load_config(args.config)
    build_bm25_index(config, force=args.force)

if __name__ == "__main__":
    main()
