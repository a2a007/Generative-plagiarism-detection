import re
from typing import List, Dict, Any

class DocumentChunker:
    """
    Sliding window chunker for source documents.
    Splits text into word chunks while tracking exact raw character offsets (start_char, end_char).
    """
    def __init__(self, words_per_chunk: int = 512, overlap_words: int = 100):
        self.words_per_chunk = words_per_chunk
        self.overlap_words = overlap_words
        self.step = max(1, words_per_chunk - overlap_words)

    def chunk_document(self, doc_id: str, raw_text: str) -> List[Dict[str, Any]]:
        """
        Chunks a raw text document into sliding window word chunks, preserving character offsets.
        """
        chunks = []
        if not raw_text.strip():
            return chunks

        # Find word boundaries and their character start/end positions in raw_text
        # Matches non-whitespace tokens
        word_spans = [(m.start(), m.end()) for m in re.finditer(r'\S+', raw_text)]
        num_words = len(word_spans)

        if num_words == 0:
            return chunks

        chunk_idx = 1
        for start_w in range(0, num_words, self.step):
            end_w = min(start_w + self.words_per_chunk, num_words)
            
            start_char = word_spans[start_w][0]
            end_char = word_spans[end_w - 1][1]
            
            chunk_text = raw_text[start_char:end_char]
            chunk_id = f"{doc_id}#chunk_{chunk_idx:04d}"

            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "text": chunk_text,
                "start_char": start_char,
                "end_char": end_char
            })

            chunk_idx += 1

            if end_w >= num_words:
                break

        return chunks
