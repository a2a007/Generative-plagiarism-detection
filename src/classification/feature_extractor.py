import re
import numpy as np
from typing import List, Dict, Any, Tuple

def get_ngrams(tokens: List[str], n: int) -> set:
    if len(tokens) < n:
        return set()
    return set(zip(*[tokens[i:] for i in range(n)]))

def jaccard_similarity(set1: set, set2: set) -> float:
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

class FeatureExtractor:
    """
    Extracts lexical, semantic, rank-based, and n-gram overlap feature vectors
    for (query, candidate_source) pairs for classification training.
    """
    FEATURE_NAMES = [
        "bm25_score",
        "dense_score",
        "rrf_score",
        "bm25_reciprocal_rank",
        "dense_reciprocal_rank",
        "fused_reciprocal_rank",
        "word_jaccard",
        "bigram_jaccard",
        "trigram_jaccard",
        "char_len_ratio"
    ]

    def extract_features_from_candidates(self, candidate_records: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        X_list = []
        y_list = []

        for rec in candidate_records:
            q_text = rec["query_text"]
            gt_source = rec["ground_truth_source"]
            q_tokens = re.findall(r'\w+', q_text.lower())
            q_bigrams = get_ngrams(q_tokens, 2)
            q_trigrams = get_ngrams(q_tokens, 3)
            q_len = len(q_text)

            candidates = rec.get("candidates", [])
            for cand in candidates:
                src_doc = cand["source_document"]
                is_positive = 1 if src_doc == gt_source else 0

                bm25_score = float(cand.get("bm25_score", 0.0))
                dense_score = float(cand.get("dense_score", 0.0))
                rrf_score = float(cand.get("rrf_score", 0.0))

                bm25_rank = cand.get("bm25_rank")
                bm25_rr = 1.0 / bm25_rank if bm25_rank else 0.0

                dense_rank = cand.get("dense_rank")
                dense_rr = 1.0 / dense_rank if dense_rank else 0.0

                fused_rank = cand.get("rank")
                fused_rr = 1.0 / fused_rank if fused_rank else 0.0

                cand_text = cand.get("best_chunk_text", "")
                c_tokens = re.findall(r'\w+', cand_text.lower())
                c_bigrams = get_ngrams(c_tokens, 2)
                c_trigrams = get_ngrams(c_tokens, 3)

                word_jaccard = jaccard_similarity(set(q_tokens), set(c_tokens))
                bigram_jaccard = jaccard_similarity(q_bigrams, c_bigrams)
                trigram_jaccard = jaccard_similarity(q_trigrams, c_trigrams)

                c_len = len(cand_text)
                char_len_ratio = min(q_len, c_len) / max(q_len, c_len, 1)

                feat = [
                    bm25_score,
                    dense_score,
                    rrf_score,
                    bm25_rr,
                    dense_rr,
                    fused_rr,
                    word_jaccard,
                    bigram_jaccard,
                    trigram_jaccard,
                    char_len_ratio
                ]

                X_list.append(feat)
                y_list.append(is_positive)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)
        return X, y, self.FEATURE_NAMES
