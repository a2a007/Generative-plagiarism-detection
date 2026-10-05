import os
import sys
import json
import argparse
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.utils.logging_utils import setup_logger
from src.retrieval.dense_retriever import DenseRetriever

logger = setup_logger("Dense_IndexBuilder")

def build_dense_index(config: dict, force: bool = False, max_chunks: int = None):
    processed_dir = Path(config["dataset"]["processed_dir"])
    indexes_dir = Path(config["dataset"]["indexes_dir"])
    indexes_dir.mkdir(parents=True, exist_ok=True)

    faiss_path = indexes_dir / "dense_faiss.index"
    meta_path = indexes_dir / "dense_metadata.pkl"
    chunks_path = processed_dir / "source_chunks.jsonl"

    if faiss_path.exists() and meta_path.exists() and not force:
        logger.info(f"Dense index already exists at {indexes_dir}. Use --force to rebuild.")
        return

    if not chunks_path.exists():
        raise FileNotFoundError(f"Source chunks file not found at: {chunks_path}. Run preprocess_pan11.py first.")

    logger.info(f"Loading source chunks from: {chunks_path}")
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
                if max_chunks and len(chunks) >= max_chunks:
                    break

    model_name = config.get("dense", {}).get("model", "sentence-transformers/all-MiniLM-L6-v2")
    top_k = config.get("dense", {}).get("top_k", 100)
    batch_size = config.get("dense", {}).get("batch_size", 256)

    logger.info(f"Loaded {len(chunks)} chunks. Building Dense FAISS index with model '{model_name}'...")
    retriever = DenseRetriever(model_name=model_name, top_k=top_k, batch_size=batch_size)
    retriever.build_index(chunks)

    logger.info(f"Saving Dense FAISS index to: {indexes_dir}")
    retriever.save_index(indexes_dir)
    logger.info("Dense FAISS Index Built Successfully!")

def main():
    parser = argparse.ArgumentParser(description="Build Dense FAISS Index for PAN-11 Source Chunks")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--force", action="store_true", help="Force rebuild index even if exists")
    parser.add_argument("--max-chunks", type=int, help="Limit maximum chunks for rapid development/testing")
    args = parser.parse_args()

    config = load_config(args.config)
    build_dense_index(config, force=args.force, max_chunks=args.max_chunks)

if __name__ == "__main__":
    main()
