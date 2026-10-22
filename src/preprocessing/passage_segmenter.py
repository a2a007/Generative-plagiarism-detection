import re
from typing import List, Dict, Any

class PassageSegmenter:
    """
    Passage-level sliding-window text segmenter for multi-source plagiarism detection.
    Splits an input document into overlapping passages while strictly preserving
    exact raw character offsets (start_char, end_char) against the original document text.
    """
    def __init__(self, window_words: int = 150, overlap_words: int = 50, min_words: int = 5):
        if window_words <= 0:
            raise ValueError("window_words must be greater than 0")
        if overlap_words < 0 or overlap_words >= window_words:
            raise ValueError("overlap_words must be >= 0 and < window_words")

        self.window_words = window_words
        self.overlap_words = overlap_words
        self.step_words = max(1, window_words - overlap_words)
        self.min_words = min_words

    def segment_text(self, raw_text: str) -> List[Dict[str, Any]]:
        """
        Segments raw_text into overlapping passage dictionaries with exact character boundaries.
        
        Returns:
            List of dicts: [
                {
                    "passage_id": "passage_0001",
                    "text": str,
                    "start_char": int,
                    "end_char": int,
                    "word_count": int
                }, ...
            ]
        """
        passages: List[Dict[str, Any]] = []
        if not raw_text or not raw_text.strip():
            return passages

        # Find word boundaries using non-whitespace token spans
        word_spans = [(m.start(), m.end()) for m in re.finditer(r'\S+', raw_text)]
        total_words = len(word_spans)

        if total_words < self.min_words:
            return passages

        # Short text handling: if total words is less than window_words, return single passage
        if total_words <= self.window_words:
            start_char = word_spans[0][0]
            end_char = word_spans[-1][1]
            passage_text = raw_text[start_char:end_char]
            passages.append({
                "passage_id": "passage_0001",
                "text": passage_text,
                "start_char": start_char,
                "end_char": end_char,
                "word_count": total_words
            })
            return passages

        passage_idx = 1
        for start_w in range(0, total_words, self.step_words):
            end_w = min(start_w + self.window_words, total_words)
            span_words = end_w - start_w

            # If the trailing slice is too small and we already produced passages, break
            if span_words < self.min_words and len(passages) > 0:
                break

            start_char = word_spans[start_w][0]
            end_char = word_spans[end_w - 1][1]
            passage_text = raw_text[start_char:end_char]

            passages.append({
                "passage_id": f"passage_{passage_idx:04d}",
                "text": passage_text,
                "start_char": start_char,
                "end_char": end_char,
                "word_count": span_words
            })

            passage_idx += 1

            if end_w >= total_words:
                break

        return passages
