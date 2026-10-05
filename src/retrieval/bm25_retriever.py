import os
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Any

def tokenize_text(text: str) -> List[str]:
    """
    Simple whitespace and alphanumeric tokenizer for BM25.
    """
    return re.findall(r'\w+', text.lower())

class BM25Retriever:
    """
    Inverted-Index BM25 Retrieval Engine for fast lexical search over large corpora.
    Calculates BM25Okapi scores in milliseconds per query.
    """
    def __init__(self, top_k: int = 100, k1: float = 1.5, b: float = 0.75):
        self.top_k = top_k
        self.k1 = k1
        self.b = b
        self.num_docs: int = 0
        self.avgdl: float = 0.0
        self.doc_lengths: List[int] = []
        self.idf: Dict[str, float] = {}
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.chunks_metadata: List[Dict[str, Any]] = []

    def build_index(self, chunks: List[Dict[str, Any]]):
        """
        Builds Inverted Index BM25 over source document chunks.
        """
        self.chunks_metadata = []
        self.doc_lengths = []
        self.num_docs = len(chunks)
        df_counts = Counter()
        self.inverted_index = defaultdict(list)

        total_length = 0
        for doc_idx, chk in enumerate(chunks):
            tokens = tokenize_text(chk["text"])
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            total_length += doc_len

            self.chunks_metadata.append({
                "chunk_id": chk["chunk_id"],
                "doc_id": chk["doc_id"],
                "start_char": chk.get("start_char", 0),
                "end_char": chk.get("end_char", 0),
                "text": chk.get("text", "")
            })

            tf_counts = Counter(tokens)
            for term, tf in tf_counts.items():
                self.inverted_index[term].append((doc_idx, tf))
                df_counts[term] += 1

        self.avgdl = total_length / self.num_docs if self.num_docs > 0 else 1.0

        # Calculate IDF (Okapi BM25 formula)
        self.idf = {}
        for term, df in df_counts.items():
            # Standard Okapi BM25 IDF formula
            idf_val = math.log(1.0 + (self.num_docs - df + 0.5) / (df + 0.5))
            self.idf[term] = max(0.0, idf_val)

    def search(self, query_text: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Searches top-K matching chunks for a query using inverted index posting lists.
        """
        if self.num_docs == 0:
            return []

        k = top_k if top_k is not None else self.top_k
        query_tokens = tokenize_text(query_text)

        if not query_tokens:
            return []

        q_counts = Counter(query_tokens)
        doc_scores: Dict[int, float] = defaultdict(float)

        for term, q_tf in q_counts.items():
            if term not in self.idf:
                continue

            term_idf = self.idf[term]
            postings = self.inverted_index[term]

            for doc_idx, tf in postings:
                doc_len = self.doc_lengths[doc_idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                score = term_idf * (tf * (self.k1 + 1.0)) / denom
                doc_scores[doc_idx] += score

        if not doc_scores:
            return []

        # Sort candidate doc indices descending by score
        top_candidates = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        results = []
        for rank, (doc_idx, score) in enumerate(top_candidates, start=1):
            meta = self.chunks_metadata[doc_idx]
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

    def save_index(self, index_path: Path):
        index_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "top_k": self.top_k,
            "k1": self.k1,
            "b": self.b,
            "num_docs": self.num_docs,
            "avgdl": self.avgdl,
            "doc_lengths": self.doc_lengths,
            "idf": self.idf,
            "inverted_index": dict(self.inverted_index),
            "chunks_metadata": self.chunks_metadata
        }
        with open(index_path, "wb") as f:
            pickle.dump(data, f)

    def load_index(self, index_path: Path):
        with open(index_path, "rb") as f:
            data = pickle.load(f)
        self.top_k = data["top_k"]
        self.k1 = data.get("k1", 1.5)
        self.b = data.get("b", 0.75)
        self.num_docs = data["num_docs"]
        self.avgdl = data["avgdl"]
        self.doc_lengths = data["doc_lengths"]
        self.idf = data["idf"]
        self.inverted_index = defaultdict(list, data["inverted_index"])
        self.chunks_metadata = data["chunks_metadata"]
