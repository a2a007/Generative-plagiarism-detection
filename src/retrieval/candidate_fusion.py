import os
import re
from typing import List, Dict, Any, Optional

def _token_jaccard(text1: str, text2: str) -> float:
    tokens1 = set(re.findall(r'\w+', text1.lower()))
    tokens2 = set(re.findall(r'\w+', text2.lower()))
    if not tokens1 or not tokens2:
        return 0.0
    return len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))

class CandidateFusion:
    """
    Reciprocal Rank Fusion (RRF) and candidate aggregation engine for lexical (BM25) and dense semantic retrieval results.
    Aggregates candidates to SOURCE DOCUMENT level and aligns the best-matching source passage chunk.
    """
    def __init__(self, rrf_k: float = 60.0):
        self.rrf_k = rrf_k

    def aggregate_chunk_results_to_doc(self, chunk_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Aggregates chunk-level retrieval results to document level by taking the best rank / highest score for each document.
        Returns doc-level ranked candidate list with doc-level rank (1-indexed).
        """
        doc_map: Dict[str, Dict[str, Any]] = {}
        for item in chunk_results:
            doc_id = item["source_document"]
            score = item["score"]
            chunk_id = item["chunk_id"]
            chunk_text = item.get("text", "")
            start_char = item.get("start_char", 0)
            end_char = item.get("end_char", 0)

            if doc_id not in doc_map or score > doc_map[doc_id]["score"]:
                doc_map[doc_id] = {
                    "source_document": doc_id,
                    "score": score,
                    "best_chunk_id": chunk_id,
                    "best_chunk_text": chunk_text,
                    "start_char": start_char,
                    "end_char": end_char
                }

        # Sort documents descending by best score
        sorted_docs = sorted(doc_map.values(), key=lambda x: x["score"], reverse=True)
        for rank, d in enumerate(sorted_docs, start=1):
            d["rank"] = rank
        return sorted_docs

    def fuse_results(
        self,
        bm25_chunk_results: List[Dict[str, Any]],
        dense_chunk_results: List[Dict[str, Any]],
        top_k: int = 100,
        query_text: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Fuses BM25 and Dense chunk retrieval results using Reciprocal Rank Fusion at document level.
        Aligns the best-matching passage chunk based on query overlap and scores.
        """
        # Aggregate to document level first
        bm25_docs = self.aggregate_chunk_results_to_doc(bm25_chunk_results)
        dense_docs = self.aggregate_chunk_results_to_doc(dense_chunk_results)

        bm25_ranks = {
            d["source_document"]: (
                d["rank"], d["score"], d["best_chunk_id"], d["best_chunk_text"], d["start_char"], d["end_char"]
            ) for d in bm25_docs
        }
        dense_ranks = {
            d["source_document"]: (
                d["rank"], d["score"], d["best_chunk_id"], d["best_chunk_text"], d["start_char"], d["end_char"]
            ) for d in dense_docs
        }

        all_doc_ids = set(bm25_ranks.keys()).union(set(dense_ranks.keys()))

        fused_candidates = []
        for doc_id in all_doc_ids:
            rrf_score = 0.0

            bm25_score = 0.0
            bm25_rank = None
            bm25_chunk_id = None
            bm25_text = ""
            bm25_start = 0
            bm25_end = 0

            if doc_id in bm25_ranks:
                bm25_rank, bm25_score, bm25_chunk_id, bm25_text, bm25_start, bm25_end = bm25_ranks[doc_id]
                rrf_score += 1.0 / (self.rrf_k + bm25_rank)

            dense_score = 0.0
            dense_rank = None
            dense_chunk_id = None
            dense_text = ""
            dense_start = 0
            dense_end = 0

            if doc_id in dense_ranks:
                dense_rank, dense_score, dense_chunk_id, dense_text, dense_start, dense_end = dense_ranks[doc_id]
                rrf_score += 1.0 / (self.rrf_k + dense_rank)

            # Intelligently select the best matching chunk between BM25 and Dense
            if bm25_chunk_id and dense_chunk_id:
                if query_text:
                    # Compare token overlap against the actual query
                    bm25_overlap = _token_jaccard(query_text, bm25_text)
                    dense_overlap = _token_jaccard(query_text, dense_text)
                    if bm25_overlap >= dense_overlap:
                        best_chunk_id = bm25_chunk_id
                        best_chunk_text = bm25_text
                        best_start = bm25_start
                        best_end = bm25_end
                    else:
                        best_chunk_id = dense_chunk_id
                        best_chunk_text = dense_text
                        best_start = dense_start
                        best_end = dense_end
                else:
                    # Default to BM25 chunk if lexical score is positive, as it captures exact phrase hits
                    if bm25_score > 0:
                        best_chunk_id = bm25_chunk_id
                        best_chunk_text = bm25_text
                        best_start = bm25_start
                        best_end = bm25_end
                    else:
                        best_chunk_id = dense_chunk_id
                        best_chunk_text = dense_text
                        best_start = dense_start
                        best_end = dense_end
            elif bm25_chunk_id:
                best_chunk_id = bm25_chunk_id
                best_chunk_text = bm25_text
                best_start = bm25_start
                best_end = bm25_end
            else:
                best_chunk_id = dense_chunk_id
                best_chunk_text = dense_text
                best_start = dense_start
                best_end = dense_end

            fused_candidates.append({
                "source_document": doc_id,
                "rrf_score": rrf_score,
                "bm25_score": bm25_score,
                "bm25_rank": bm25_rank,
                "dense_score": dense_score,
                "dense_rank": dense_rank,
                "best_chunk_id": best_chunk_id,
                "best_chunk_text": best_chunk_text,
                "start_char": best_start,
                "end_char": best_end
            })

        # Sort descending by RRF score
        fused_candidates.sort(key=lambda x: x["rrf_score"], reverse=True)
        for rank, item in enumerate(fused_candidates, start=1):
            item["rank"] = rank

        return fused_candidates[:top_k]
