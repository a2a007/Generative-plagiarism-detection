import json
from pathlib import Path
from typing import List, Dict, Any
from src.preprocessing.pan_parser import PlagiarismAnnotation
from src.preprocessing.document_loader import DocumentLoader

class GroundTruthExtractor:
    """
    Extracts and prepares ground truth JSONL records and retrieval query records from PAN-11 annotations.
    """
    def __init__(self, document_loader: DocumentLoader):
        self.doc_loader = document_loader

    def build_ground_truth_and_queries(self, annotations: List[PlagiarismAnnotation]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        gt_records = []
        query_records = []
        susp_text_cache: Dict[str, str] = {}

        for idx, ann in enumerate(annotations):
            query_id = f"query_{idx+1:06d}"

            # 1. Ground truth record
            gt_rec = {
                "query_id": query_id,
                "suspicious_id": ann.suspicious_id,
                "suspicious_offset": ann.suspicious_offset,
                "suspicious_length": ann.suspicious_length,
                "source_id": ann.source_id,
                "source_offset": ann.source_offset,
                "source_length": ann.source_length,
                "obfuscation": ann.obfuscation,
                "type": ann.type,
                "this_language": ann.this_language,
                "source_language": ann.source_language
            }
            gt_records.append(gt_rec)

            # 2. Extract query text from suspicious document WITHOUT using source ground truth
            if ann.suspicious_id not in susp_text_cache:
                file_path = self.doc_loader.get_suspicious_file_path(ann.suspicious_id)
                if file_path and file_path.exists():
                    susp_text_cache[ann.suspicious_id] = self.doc_loader.load_document_raw(file_path)
                else:
                    susp_text_cache[ann.suspicious_id] = ""

            raw_text = susp_text_cache[ann.suspicious_id]
            start = max(0, ann.suspicious_offset)
            end = min(len(raw_text), ann.suspicious_offset + ann.suspicious_length)
            query_text = raw_text[start:end]

            query_rec = {
                "query_id": query_id,
                "suspicious_id": ann.suspicious_id,
                "suspicious_offset": ann.suspicious_offset,
                "suspicious_length": ann.suspicious_length,
                "query_text": query_text,
                "obfuscation": ann.obfuscation,
                "type": ann.type
            }
            query_records.append(query_rec)

        return gt_records, query_records

    def save_jsonl(self, records: List[Dict[str, Any]], output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
