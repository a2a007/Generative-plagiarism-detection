import os
import json
import pickle
import torch
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class DenseRetriever:
    """
    Dense Semantic Retrieval Engine using Sentence Transformers and FAISS.
    Configured for fast multi-threaded CPU inference.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", top_k: int = 100, batch_size: int = 256):
        self.model_name = model_name
        self.top_k = top_k
        self.batch_size = batch_size
        self.model: SentenceTransformer = None
        self.index: faiss.IndexFlatIP = None
        self.chunks_metadata: List[Dict[str, Any]] = []

    def _load_model(self):
        if self.model is None:
            # Enable optimal CPU threading
            cpu_count = os.cpu_count() or 4
            torch.set_num_threads(cpu_count)
            self.model = SentenceTransformer(self.model_name)

    def build_index(self, chunks: List[Dict[str, Any]]):
        """
        Encodes source chunks and builds a normalized FAISS IndexFlatIP index.
        """
        self._load_model()
        self.chunks_metadata = []
        texts = []

        for chk in chunks:
            texts.append(chk["text"])
            self.chunks_metadata.append({
                "chunk_id": chk["chunk_id"],
                "doc_id": chk["doc_id"],
                "start_char": chk.get("start_char", 0),
                "end_char": chk.get("end_char", 0),
                "text": chk.get("text", "")
            })

        logger_msg = f"Encoding {len(texts)} chunks with batch size {self.batch_size}..."
        print(logger_msg)

        # Encode with SentenceTransformer
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        dimension = embeddings.shape[1] if embeddings.shape[0] > 0 else 384
        self.index = faiss.IndexFlatIP(dimension)
        if embeddings.shape[0] > 0:
            self.index.add(embeddings)

    def search(self, query_text: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Encodes query text and searches FAISS index for top-K matching chunks.
        """
        if self.index is None:
            raise ValueError("FAISS index has not been built or loaded.")
            
        self._load_model()

        k = top_k if top_k is not None else self.top_k
        query_emb = self.model.encode([query_text], show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True).astype("float32")

        scores, indices = self.index.search(query_emb, k)
        
        results = []
        if indices.shape[1] > 0:
            row_indices = indices[0]
            row_scores = scores[0]

            for rank, (idx, score) in enumerate(zip(row_indices, row_scores), start=1):
                if idx < 0 or idx >= len(self.chunks_metadata):
                    continue
                meta = self.chunks_metadata[idx]
                results.append({
                    "chunk_id": meta["chunk_id"],
                    "source_document": meta["doc_id"],
                    "score": float(score),
                    "rank": rank,
                    "start_char": meta["start_char"],
                    "end_char": meta["end_char"],
                    "text": meta["text"]
                })

        return results

    def save_index(self, index_dir: Path):
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss_path = index_dir / "dense_faiss.index"
        meta_path = index_dir / "dense_metadata.pkl"

        faiss.write_index(self.index, str(faiss_path))
        data = {
            "model_name": self.model_name,
            "top_k": self.top_k,
            "batch_size": self.batch_size,
            "chunks_metadata": self.chunks_metadata
        }
        with open(meta_path, "wb") as f:
            pickle.dump(data, f)

    def load_index(self, index_dir: Path):
        faiss_path = index_dir / "dense_faiss.index"
        meta_path = index_dir / "dense_metadata.pkl"

        if not faiss_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Dense index files missing at: {index_dir}")

        self.index = faiss.read_index(str(faiss_path))
        with open(meta_path, "rb") as f:
            data = pickle.load(f)
        self.model_name = data["model_name"]
        self.top_k = data["top_k"]
        self.batch_size = data["batch_size"]
        self.chunks_metadata = data["chunks_metadata"]
