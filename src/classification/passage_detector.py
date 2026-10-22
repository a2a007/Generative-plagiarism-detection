import re
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from src.preprocessing.passage_segmenter import PassageSegmenter
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.candidate_fusion import CandidateFusion
from src.classification.feature_extractor import FeatureExtractor, get_ngrams, jaccard_similarity
from src.utils.logging_utils import setup_logger

logger = setup_logger("PassagePlagiarismDetector")

def extract_best_matching_snippet(query_text: str, candidate_text: str, max_chars: int = 400) -> Tuple[str, int, int]:
    """
    Finds the exact contiguous sentence region inside candidate_text that maximizes
    query token coverage and density, returning the aligned snippet and character offsets.
    """
    if not candidate_text:
        return "", 0, 0

    if len(candidate_text) <= max_chars:
        return candidate_text.strip(), 0, len(candidate_text)

    q_tokens = set(re.findall(r'\w+', query_text.lower()))
    if not q_tokens:
        return candidate_text[:max_chars].strip() + "...", 0, min(max_chars, len(candidate_text))

    sentence_matches = list(re.finditer(r'[^.!?\n]+[.!?\n]*', candidate_text))
    if not sentence_matches:
        return candidate_text[:max_chars].strip() + "...", 0, min(max_chars, len(candidate_text))

    sent_tokens = [set(re.findall(r'\w+', m.group().lower())).intersection(q_tokens) for m in sentence_matches]

    best_score = -1.0
    best_start = 0
    best_end = min(max_chars, len(candidate_text))

    for i in range(len(sentence_matches)):
        if not sent_tokens[i]:
            continue
        covered_tokens = set()
        for j in range(i, len(sentence_matches)):
            start_char = sentence_matches[i].start()
            end_char = sentence_matches[j].end()
            span_len = end_char - start_char

            if span_len > max_chars and j > i:
                break

            covered_tokens.update(sent_tokens[j])
            overlap_count = len(covered_tokens)
            density = overlap_count / max(span_len, 1)
            composite_score = overlap_count * 1000.0 + density * 100.0

            if composite_score > best_score:
                best_score = composite_score
                best_start = start_char
                best_end = end_char

    snippet = candidate_text[best_start:best_end].strip()
    prefix = "..." if best_start > 0 else ""
    suffix = "..." if best_end < len(candidate_text) else ""
    return f"{prefix}{snippet}{suffix}", best_start, best_end


def align_passage_to_candidate(query_text: str, candidate_text: str) -> Tuple[int, int, str, bool]:
    """
    Finds the exact matching character span [rel_start, rel_end] of query_text inside candidate_text.
    Uses n-gram token matching (from 5-grams down to 2-grams) to identify the true extent of the match
    without artificial length capping, returning the aligned relative offsets, formatted snippet, and alignment status.
    """
    if not candidate_text:
        return 0, 0, "", False
    q_matches = list(re.finditer(r'\b\w+\b', query_text))
    q_words = [m.group().lower() for m in q_matches]
    c_matches = list(re.finditer(r'\b\w+\b', candidate_text))
    c_words = [m.group().lower() for m in c_matches]

    if not q_words or not c_words:
        snippet = candidate_text[:300].strip()
        suffix = "..." if len(candidate_text) > 300 else ""
        return 0, min(len(candidate_text), 400), f"{snippet}{suffix}", False

    for n in (5, 4, 3):
        if len(q_words) >= n:
            q_ngrams = set(tuple(q_words[i:i+n]) for i in range(len(q_words) - n + 1))
            matching_indices = [j for j in range(len(c_words) - n + 1) if tuple(c_words[j:j+n]) in q_ngrams]
            if matching_indices:
                first_c = min(matching_indices)
                last_c = max(matching_indices) + n - 1
                rel_start = c_matches[first_c].start()
                rel_end = c_matches[last_c].end()
                snippet = candidate_text[rel_start:rel_end].strip()
                prefix = "..." if rel_start > 0 else ""
                suffix = "..." if rel_end < len(candidate_text) else ""
                disp = snippet if len(snippet) <= 300 else snippet[:300] + "..."
                return rel_start, rel_end, f"{prefix}{disp}{suffix}", True

    # Fallback to sentence coverage
    disp, s, e = extract_best_matching_snippet(query_text, candidate_text, max_chars=400)
    return s, e, disp, False


def merge_intervals(intervals: List[Tuple[int, int]], max_gap: int = 5) -> List[Tuple[int, int]]:
    """
    Merges overlapping or touching (start, end) intervals into disjoint continuous spans.
    max_gap controls the maximum whitespace/newline gap between adjacent matching spans
    that should be merged into a single continuous region.
    """
    if not intervals:
        return []
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]
    for current in sorted_intervals[1:]:
        prev_start, prev_end = merged[-1]
        curr_start, curr_end = current
        if curr_start <= prev_end + max_gap:
            merged[-1] = (prev_start, max(prev_end, curr_end))
        else:
            merged.append(current)
    return merged


class PassagePlagiarismDetector:
    """
    Passage-Level Multi-Source Plagiarism Detection Engine.
    Segments input documents into overlapping passages, evaluates each passage independently
    through lexical-semantic retrieval and classifier inference, and aggregates detections by source.
    """
    def __init__(
        self,
        config: Dict[str, Any],
        model_pickle_path: str = "outputs/trained_plagiarism_classifier.pkl",
        confidence_threshold: float = 0.50,
        window_words: int = 100,
        overlap_words: int = 30,
        min_words: int = 5,
        split_paragraphs: bool = True
    ):
        self.config = config
        self.confidence_threshold = confidence_threshold
        self.indexes_dir = Path(config["dataset"]["indexes_dir"])
        self.bm25_index_path = self.indexes_dir / "bm25_index.pkl"
        self.model_pickle_path = Path(model_pickle_path)

        self.segmenter = PassageSegmenter(
            window_words=window_words,
            overlap_words=overlap_words,
            min_words=min_words,
            split_paragraphs=split_paragraphs
        )

        self._load_retrievers()
        self._load_classifier()

    def _load_retrievers(self):
        logger.info(f"Loading BM25 index from {self.bm25_index_path}...")
        self.bm25_retriever = BM25Retriever(top_k=50)
        self.bm25_retriever.load_index(self.bm25_index_path)

        dense_model_name = self.config.get("dense", {}).get("model", "sentence-transformers/all-MiniLM-L6-v2")
        logger.info(f"Loading Dense retriever with model '{dense_model_name}'...")
        self.dense_retriever = DenseRetriever(model_name=dense_model_name, top_k=50)
        self.dense_retriever.load_index(self.indexes_dir)

        rrf_k = self.config.get("fusion", {}).get("k", 60)
        self.fusion_engine = CandidateFusion(rrf_k=rrf_k)

    def _load_classifier(self):
        logger.info(f"Loading trained classifier from {self.model_pickle_path}...")
        with open(self.model_pickle_path, "rb") as f:
            data = pickle.load(f)
        self.classifier = data["model"]
        self.scaler = data["scaler"]
        self.feature_names = data.get("feature_names", FeatureExtractor.FEATURE_NAMES)

    def extract_features_for_pair(
        self,
        query_text: str,
        cand: Dict[str, Any],
        query_emb: Optional[np.ndarray] = None
    ) -> Tuple[List[float], Dict[str, Any]]:
        """
        Calculates the exact 10 feature values for a (passage, candidate) pair.
        Order matches training script:
        [bm25_score, dense_score, rrf_score, bm25_rr, dense_rr, fused_rr,
         word_jaccard, bigram_jaccard, trigram_jaccard, char_len_ratio]
        """
        q_tokens = re.findall(r'\w+', query_text.lower())
        q_bigrams = get_ngrams(q_tokens, 2)
        q_trigrams = get_ngrams(q_tokens, 3)
        q_len = len(query_text)

        cand_text = cand.get("best_chunk_text", "")
        c_tokens = re.findall(r'\w+', cand_text.lower())
        c_bigrams = get_ngrams(c_tokens, 2)
        c_trigrams = get_ngrams(c_tokens, 3)
        c_len = len(cand_text)

        word_jaccard = jaccard_similarity(set(q_tokens), set(c_tokens))
        bigram_jaccard = jaccard_similarity(q_bigrams, c_bigrams)
        trigram_jaccard = jaccard_similarity(q_trigrams, c_trigrams)
        char_len_ratio = min(q_len, c_len) / max(q_len, c_len, 1)

        bm25_score = float(cand.get("bm25_score", 0.0))
        dense_score = float(cand.get("dense_score", 0.0))

        # Dynamic exact pairwise cosine similarity if model is loaded and text exists
        if cand_text and query_emb is not None:
            try:
                c_emb = self.dense_retriever.model.encode(
                    [cand_text],
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                exact_sim = float(np.dot(query_emb[0], c_emb[0]))
                dense_score = max(dense_score, exact_sim)
            except Exception:
                pass

        rrf_score = float(cand.get("rrf_score", 0.0))

        bm25_rank = cand.get("bm25_rank")
        bm25_rr = 1.0 / bm25_rank if bm25_rank else 0.0

        dense_rank = cand.get("dense_rank")
        dense_rr = 1.0 / dense_rank if dense_rank else 0.0

        fused_rank = cand.get("rank")
        fused_rr = 1.0 / fused_rank if fused_rank else 0.0

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

        meta = {
            "bm25_score": bm25_score,
            "dense_score": dense_score,
            "rrf_score": rrf_score,
            "bm25_rank": bm25_rank,
            "dense_rank": dense_rank,
            "fused_rank": fused_rank,
            "word_jaccard": word_jaccard,
            "bigram_jaccard": bigram_jaccard,
            "trigram_jaccard": trigram_jaccard,
            "char_len_ratio": char_len_ratio
        }

        return feat, meta

    def detect_passage(
        self,
        passage: Dict[str, Any],
        top_k: int = 5,
        debug: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Evaluates a single passage against candidate source documents.
        Returns ranked candidate evaluations for this passage.
        """
        q_text = passage["text"]
        if not q_text.strip():
            return []

        # 1. BM25 Search
        bm25_chunk_results = self.bm25_retriever.search(q_text, top_k=top_k * 3)

        # 2. Dense Search
        dense_chunk_results = self.dense_retriever.search(q_text, top_k=top_k * 3)

        # 3. Candidate Fusion
        fused_candidates = self.fusion_engine.fuse_results(
            bm25_chunk_results,
            dense_chunk_results,
            top_k=top_k * 2,
            query_text=q_text
        )

        # Encode query once for pairwise similarity
        self.dense_retriever._load_model()
        q_emb = self.dense_retriever.model.encode(
            [q_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        passage_evaluations = []

        for cand in fused_candidates[:top_k]:
            src_doc = cand["source_document"]
            cand_text = cand.get("best_chunk_text", "")
            chunk_start = cand.get("start_char", 0)
            chunk_end = cand.get("end_char", 0)

            # Feature extraction
            feat_unscaled, meta = self.extract_features_for_pair(q_text, cand, query_emb=q_emb)
            feat_scaled = self.scaler.transform([feat_unscaled])

            # Prediction with explicit positive class selection
            if hasattr(self.classifier, "predict_proba"):
                probs = self.classifier.predict_proba(feat_scaled)[0]
                classes = getattr(self.classifier, "classes_", [0, 1])
                pos_idx = list(classes).index(1) if 1 in classes else (1 if len(classes) > 1 else 0)
                prob = float(probs[pos_idx])
            else:
                prob = float(self.classifier.predict(feat_scaled)[0])

            has_textual_evidence = (
                meta["trigram_jaccard"] >= 0.01 or
                meta["bigram_jaccard"] >= 0.03 or
                (meta["dense_score"] >= 0.50 and meta["word_jaccard"] >= 0.12) or
                meta["word_jaccard"] >= 0.15
            )
            is_match = (prob >= self.confidence_threshold) and has_textual_evidence

            # Align snippet and calculate exact offsets
            rel_start, rel_end, aligned_snippet, exact_aligned = align_passage_to_candidate(q_text, cand_text)
            exact_source_start = chunk_start + rel_start
            exact_source_end = chunk_start + rel_end

            eval_record = {
                "passage_id": passage["passage_id"],
                "input_passage_offsets": [passage["start_char"], passage["end_char"]],
                "passage_word_count": passage["word_count"],
                "source_document": src_doc,
                "plagiarism_decision": "PLAGIARISM_MATCH" if is_match else "NO_MATCH",
                "confidence_score": round(prob, 4),
                "exact_alignment_verified": exact_aligned,
                "fused_rank": cand.get("rank"),
                "bm25_rank": meta["bm25_rank"],
                "dense_rank": meta["dense_rank"],
                "matched_chunk_id": cand.get("best_chunk_id"),
                "source_passage_offsets": [exact_source_start, exact_source_end],
                "bm25_score": round(meta["bm25_score"], 4),
                "dense_similarity": round(meta["dense_score"], 4),
                "rrf_score": round(meta["rrf_score"], 4),
                "word_jaccard": round(meta["word_jaccard"], 4),
                "bigram_jaccard": round(meta["bigram_jaccard"], 4),
                "trigram_jaccard": round(meta["trigram_jaccard"], 4),
                "matched_source_snippet": aligned_snippet,
                "input_passage_text": q_text
            }

            if debug:
                eval_record["debug_features"] = {
                    "unscaled": [float(round(x, 4)) for x in feat_unscaled],
                    "scaled": [float(round(x, 4)) for x in feat_scaled[0]],
                    "feature_names": [str(f) for f in self.feature_names],
                    "model_type": str(type(self.classifier)),
                    "classes": [int(x) for x in getattr(self.classifier, "classes_", [0, 1])],
                    "bm25_rank": meta["bm25_rank"],
                    "dense_rank": meta["dense_rank"],
                    "fused_rank": cand.get("rank"),
                    "has_textual_evidence": bool(has_textual_evidence)
                }

            passage_evaluations.append(eval_record)

        return passage_evaluations

    def detect_document(
        self,
        raw_text: str,
        doc_name: str = "Input_Document",
        top_k: int = 5,
        debug: bool = False
    ) -> Dict[str, Any]:
        """
        Executes full passage-level multi-source plagiarism detection on raw_text.
        Segments document, evaluates each passage, aggregates results by source document,
        and produces both source-level summary and passage-level details.
        """
        passages = self.segmenter.segment_text(raw_text)
        total_chars = len(raw_text)
        total_words = len(re.findall(r'\S+', raw_text))

        all_passage_evals = []
        positive_detections_by_source: Dict[str, List[Dict[str, Any]]] = {}
        all_retrieved_sources: Dict[str, Dict[str, Any]] = {}

        for passage in passages:
            p_evals = self.detect_passage(passage, top_k=top_k, debug=debug)
            all_passage_evals.extend(p_evals)

            for ev in p_evals:
                src_doc = ev["source_document"]

                # Track overall retrieval data for every source
                if src_doc not in all_retrieved_sources:
                    all_retrieved_sources[src_doc] = {
                        "source_document": src_doc,
                        "highest_confidence": ev["confidence_score"],
                        "best_bm25_score": ev["bm25_score"],
                        "best_dense_similarity": ev["dense_similarity"],
                        "best_rrf_score": ev["rrf_score"],
                        "best_fused_rank": ev["fused_rank"],
                        "best_snippet": ev["matched_source_snippet"]
                    }
                else:
                    rec = all_retrieved_sources[src_doc]
                    if ev["confidence_score"] > rec["highest_confidence"]:
                        rec["highest_confidence"] = ev["confidence_score"]
                        rec["best_snippet"] = ev["matched_source_snippet"]
                    rec["best_bm25_score"] = max(rec["best_bm25_score"], ev["bm25_score"])
                    rec["best_dense_similarity"] = max(rec["best_dense_similarity"], ev["dense_similarity"])
                    rec["best_rrf_score"] = max(rec["best_rrf_score"], ev["rrf_score"])
                    rec["best_fused_rank"] = min(rec["best_fused_rank"], ev["fused_rank"])

                # Group confirmed matches
                if ev["plagiarism_decision"] == "PLAGIARISM_MATCH":
                    if src_doc not in positive_detections_by_source:
                        positive_detections_by_source[src_doc] = []
                    positive_detections_by_source[src_doc].append(ev)

        # Multi-source aggregation
        detected_sources = []
        all_confirmed_input_intervals = []

        for src_doc, hits in positive_detections_by_source.items():
            max_conf = max(h["confidence_score"] for h in hits)
            peak_hit = max(hits, key=lambda h: h["confidence_score"])

            input_intervals = [tuple(h["input_passage_offsets"]) for h in hits]
            source_intervals = [tuple(h["source_passage_offsets"]) for h in hits]
            all_confirmed_input_intervals.extend(input_intervals)

            merged_input_spans = merge_intervals(input_intervals, max_gap=5)
            merged_source_spans = merge_intervals(source_intervals, max_gap=5)

            detected_sources.append({
                "source_document": src_doc,
                "source_level_decision": "PLAGIARISM_MATCH",
                "highest_passage_confidence": max_conf,
                "evaluated_windows_count": len(passages),
                "confirmed_passage_detections": len(hits),
                "merged_input_regions_count": len(merged_input_spans),
                "detected_passages_count": len(hits),
                "exact_alignment_verified": any(h.get("exact_alignment_verified", False) for h in hits),
                "merged_input_offsets": [[s, e] for s, e in merged_input_spans],
                "merged_source_offsets": [[s, e] for s, e in merged_source_spans],
                "best_bm25_score": max(h["bm25_score"] for h in hits),
                "best_dense_similarity": max(h["dense_similarity"] for h in hits),
                "best_rrf_score": max(h["rrf_score"] for h in hits),
                "representative_snippet": peak_hit["matched_source_snippet"],
                "matching_passage_details": hits
            })

        # Sort detected sources descending by confidence
        detected_sources.sort(key=lambda s: s["highest_passage_confidence"], reverse=True)

        overall_merged_input_regions = merge_intervals(all_confirmed_input_intervals, max_gap=5)

        # Also compile non-matching candidate sources
        non_matching_sources = []
        for src_doc, rec in all_retrieved_sources.items():
            if src_doc not in positive_detections_by_source:
                non_matching_sources.append({
                    "source_document": src_doc,
                    "source_level_decision": "NO_MATCH",
                    "highest_passage_confidence": rec["highest_confidence"],
                    "best_bm25_score": rec["best_bm25_score"],
                    "best_dense_similarity": rec["best_dense_similarity"],
                    "best_rrf_score": rec["best_rrf_score"],
                    "best_fused_rank": rec["best_fused_rank"],
                    "representative_snippet": rec["best_snippet"]
                })

        non_matching_sources.sort(key=lambda s: s["highest_passage_confidence"], reverse=True)

        has_plagiarism = len(detected_sources) > 0
        total_confirmed_windows = sum(len(hits) for hits in positive_detections_by_source.values())

        report = {
            "input_document_name": doc_name,
            "input_text_length_chars": total_chars,
            "input_text_word_count": total_words,
            "total_evaluated_windows": len(passages),
            "confirmed_passage_detections": total_confirmed_windows,
            "merged_input_regions_count": len(overall_merged_input_regions),
            "unique_source_documents_count": len(detected_sources),
            "overall_merged_input_offsets": [[s, e] for s, e in overall_merged_input_regions],
            "total_passages_evaluated": len(passages),
            "detected_sources_count": len(detected_sources),
            "overall_verdict": f"PLAGIARISM DETECTED ({len(detected_sources)} source{'s' if len(detected_sources) != 1 else ''})" if has_plagiarism else "NO SIGNIFICANT PLAGIARISM DETECTED",
            "confirmed_detected_sources": detected_sources,
            "retrieved_non_matching_sources": non_matching_sources,
            "passage_evaluations_count": len(all_passage_evals),
            "all_passage_evaluations": all_passage_evals
        }

        return report
