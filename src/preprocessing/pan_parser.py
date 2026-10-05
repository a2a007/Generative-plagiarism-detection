import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict

@dataclass
class PlagiarismAnnotation:
    suspicious_id: str
    suspicious_offset: int
    suspicious_length: int
    source_id: str
    source_offset: int
    source_length: int
    obfuscation: str
    type: str
    this_language: str
    source_language: str
    xml_path: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PANParser:
    """
    Parser for PAN-11 Plagiarism Corpus XML annotation files.
    """
    def __init__(self, external_corpus_root: str):
        self.external_corpus_root = Path(external_corpus_root)
        self.suspicious_dir = self.external_corpus_root / "suspicious-document"
        self.source_dir = self.external_corpus_root / "source-document"

    def parse_xml_file(self, xml_path: Path) -> Tuple[List[PlagiarismAnnotation], List[Dict[str, Any]]]:
        """
        Parses a single PAN XML annotation file.
        Returns:
            (annotations_list, errors_list)
        """
        annotations = []
        errors = []

        if not xml_path.exists():
            errors.append({
                "file": str(xml_path),
                "error": "File does not exist"
            })
            return annotations, errors

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError as pe:
            errors.append({
                "file": str(xml_path),
                "error": f"XML parse error: {str(pe)}"
            })
            return annotations, errors
        except Exception as e:
            errors.append({
                "file": str(xml_path),
                "error": f"Unexpected error reading XML: {str(e)}"
            })
            return annotations, errors

        # Document reference
        doc_ref = root.attrib.get("reference")
        if not doc_ref:
            # Fallback to XML filename with .txt extension
            doc_ref = xml_path.stem + ".txt"

        for feature in root.findall("feature"):
            if feature.attrib.get("name") == "plagiarism":
                attrib = feature.attrib
                source_ref = attrib.get("source_reference")
                
                if not source_ref:
                    errors.append({
                        "file": str(xml_path),
                        "error": "Incomplete annotation: missing source_reference attribute",
                        "attributes": dict(attrib)
                    })
                    continue

                try:
                    this_offset = int(attrib.get("this_offset", -1))
                    this_length = int(attrib.get("this_length", -1))
                    source_offset = int(attrib.get("source_offset", -1))
                    source_length = int(attrib.get("source_length", -1))
                except ValueError as ve:
                    errors.append({
                        "file": str(xml_path),
                        "error": f"Invalid offset/length integer conversion: {str(ve)}",
                        "attributes": dict(attrib)
                    })
                    continue

                if this_offset < 0 or this_length < 0 or source_offset < 0 or source_length < 0:
                    errors.append({
                        "file": str(xml_path),
                        "error": "Negative character offset or length detected",
                        "attributes": dict(attrib)
                    })
                    continue

                annotation = PlagiarismAnnotation(
                    suspicious_id=doc_ref,
                    suspicious_offset=this_offset,
                    suspicious_length=this_length,
                    source_id=source_ref,
                    source_offset=source_offset,
                    source_length=source_length,
                    obfuscation=attrib.get("obfuscation", "unknown"),
                    type=attrib.get("type", "unknown"),
                    this_language=attrib.get("this_language", "en"),
                    source_language=attrib.get("source_language", "en"),
                    xml_path=str(xml_path)
                )
                annotations.append(annotation)

        return annotations, errors

    def parse_directory(self, parts: Optional[List[str]] = None) -> Tuple[List[PlagiarismAnnotation], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Recursively scans suspicious-document directory (or specific part subdirectories).
        Returns:
            (all_annotations, all_errors, summary_stats)
        """
        all_annotations = []
        all_errors = []
        
        xml_files = []
        if parts:
            for part in parts:
                part_dir = self.suspicious_dir / part if not part.startswith("part") else self.suspicious_dir / part
                if part_dir.exists():
                    xml_files.extend(list(part_dir.glob("*.xml")))
        else:
            xml_files = list(self.suspicious_dir.rglob("*.xml"))

        xml_files.sort()
        suspicious_docs_with_plagiarism = set()
        total_xml_files = len(xml_files)

        for xml_file in xml_files:
            annots, errs = self.parse_xml_file(xml_file)
            if errs:
                all_errors.extend(errs)
            if annots:
                all_annotations.extend(annots)
                for a in annots:
                    suspicious_docs_with_plagiarism.add(a.suspicious_id)

        stats = {
            "total_xml_files": total_xml_files,
            "total_plagiarism_cases": len(all_annotations),
            "suspicious_docs_with_plagiarism_count": len(suspicious_docs_with_plagiarism),
            "error_count": len(all_errors)
        }

        return all_annotations, all_errors, stats
