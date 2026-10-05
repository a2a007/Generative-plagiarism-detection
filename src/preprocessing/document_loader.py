import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class DocumentLoader:
    """
    Loader for raw PAN-11 TXT documents preserving raw character alignment and UTF-8 encoding.
    """
    def __init__(self, external_corpus_root: str):
        self.external_corpus_root = Path(external_corpus_root)
        self.source_dir = self.external_corpus_root / "source-document"
        self.suspicious_dir = self.external_corpus_root / "suspicious-document"
        self._source_index: Optional[Dict[str, Path]] = None
        self._suspicious_index: Optional[Dict[str, Path]] = None

    def _build_index(self, base_dir: Path) -> Dict[str, Path]:
        mapping = {}
        if base_dir.exists():
            for f in base_dir.rglob("*.txt"):
                mapping[f.name] = f
        return mapping

    def get_source_file_path(self, doc_id: str) -> Optional[Path]:
        if self._source_index is None:
            self._source_index = self._build_index(self.source_dir)
        return self._source_index.get(doc_id)

    def get_suspicious_file_path(self, doc_id: str) -> Optional[Path]:
        if self._suspicious_index is None:
            self._suspicious_index = self._build_index(self.suspicious_dir)
        return self._suspicious_index.get(doc_id)

    def load_document_raw(self, file_path: Path) -> str:
        """
        Reads raw document text preserving exact character layout.
        """
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    def extract_passage(self, file_path: Path, offset: int, length: int) -> str:
        """
        Extracts raw character substring using PAN offsets.
        """
        text = self.load_document_raw(file_path)
        start = max(0, offset)
        end = min(len(text), offset + length)
        return text[start:end]

    def load_all_source_documents(self, parts: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Loads source documents for specified parts or all if None.
        Returns: {doc_id: raw_text}
        """
        docs = {}
        if parts:
            for part in parts:
                p_dir = self.source_dir / (f"part{part}" if not part.startswith("part") else part)
                if p_dir.exists():
                    for txt_file in p_dir.glob("*.txt"):
                        docs[txt_file.name] = self.load_document_raw(txt_file)
        else:
            if self._source_index is None:
                self._source_index = self._build_index(self.source_dir)
            for doc_id, path in self._source_index.items():
                docs[doc_id] = self.load_document_raw(path)
        return docs
