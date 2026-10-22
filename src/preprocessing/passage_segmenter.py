import re
from typing import List, Dict, Any, Tuple

class PassageSegmenter:
    """
    Passage-level text segmenter for multi-source plagiarism detection.
    Splits an input document into structural passages (paragraphs or sliding sentence/word windows)
    while strictly preserving exact raw character offsets (start_char, end_char) against the original document text.
    """
    def __init__(
        self,
        window_words: int = 100,
        overlap_words: int = 30,
        min_words: int = 5,
        split_paragraphs: bool = True
    ):
        if window_words <= 0:
            raise ValueError("window_words must be greater than 0")
        if overlap_words < 0 or overlap_words >= window_words:
            raise ValueError("overlap_words must be >= 0 and < window_words")

        self.window_words = window_words
        self.overlap_words = overlap_words
        self.step_words = max(1, window_words - overlap_words)
        self.min_words = min_words
        self.split_paragraphs = split_paragraphs

    def _segment_span(self, raw_text: str, span_start: int, span_end: int, start_idx: int) -> Tuple[List[Dict[str, Any]], int]:
        """
        Segments a specific substring span (e.g. a paragraph) into word sliding windows,
        preserving absolute character offsets relative to raw_text.
        """
        span_text = raw_text[span_start:span_end]
        word_spans = [(span_start + m.start(), span_start + m.end()) for m in re.finditer(r'\S+', span_text)]
        total_words = len(word_spans)

        if total_words < self.min_words:
            return [], start_idx

        passages = []
        if total_words <= self.window_words:
            start_char = word_spans[0][0]
            end_char = word_spans[-1][1]
            passages.append({
                "passage_id": f"passage_{start_idx:04d}",
                "text": raw_text[start_char:end_char],
                "start_char": start_char,
                "end_char": end_char,
                "word_count": total_words
            })
            return passages, start_idx + 1

        curr_idx = start_idx
        for start_w in range(0, total_words, self.step_words):
            end_w = min(start_w + self.window_words, total_words)
            span_count = end_w - start_w
            if span_count < self.min_words and len(passages) > 0:
                break
            start_char = word_spans[start_w][0]
            end_char = word_spans[end_w - 1][1]
            passages.append({
                "passage_id": f"passage_{curr_idx:04d}",
                "text": raw_text[start_char:end_char],
                "start_char": start_char,
                "end_char": end_char,
                "word_count": span_count
            })
            curr_idx += 1
            if end_w >= total_words:
                break

        return passages, curr_idx

    def segment_text(self, raw_text: str) -> List[Dict[str, Any]]:
        """
        Segments raw_text into passage dictionaries.
        If split_paragraphs is True and multiple paragraphs exist, each paragraph is segmented independently,
        preventing copied text from different sources in separate paragraphs from contaminating each other.
        """
        if not raw_text or not raw_text.strip():
            return []

        # Find paragraph blocks separated by blank lines (LF or CRLF)
        if self.split_paragraphs:
            para_matches = [m for m in re.finditer(r'\S.*?(?=\r?\n\s*\r?\n|$)', raw_text, re.DOTALL) if m.group().strip()]
            if len(para_matches) > 1:
                all_passages = []
                idx = 1
                for pm in para_matches:
                    p_start = pm.start()
                    p_end = pm.end()
                    p_list, idx = self._segment_span(raw_text, p_start, p_end, idx)
                    all_passages.extend(p_list)
                if all_passages:
                    return all_passages

        # Fallback to single block segmentation if no paragraph breaks or split_paragraphs=False
        passages, _ = self._segment_span(raw_text, 0, len(raw_text), 1)
        return passages
