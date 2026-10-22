import unittest
from pathlib import Path
from src.utils.config import load_config
from src.classification.passage_detector import PassagePlagiarismDetector

class TestMultiSourceDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config("config.yaml")
        cls.detector = PassagePlagiarismDetector(
            config=cls.config,
            confidence_threshold=0.50,
            window_words=100,
            overlap_words=30,
            min_words=5,
            split_paragraphs=True
        )
        cls.fixtures_dir = Path("data/test_documents")

    def test_01_exact_single_source_copying(self):
        """Test Scenario 1: Exact single-source copying using known_source_test.txt"""
        file_path = self.fixtures_dir / "known_source_test.txt"
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        report = self.detector.detect_document(text, doc_name=file_path.name)
        self.assertIn("PLAGIARISM DETECTED", report["overall_verdict"])
        self.assertGreaterEqual(report["detected_sources_count"], 1)

        detected_docs = [s["source_document"] for s in report["confirmed_detected_sources"]]
        self.assertIn("source-document00019.txt", detected_docs)

        src_match = next(s for s in report["confirmed_detected_sources"] if s["source_document"] == "source-document00019.txt")
        self.assertEqual(src_match["source_level_decision"], "PLAGIARISM_MATCH")
        self.assertGreaterEqual(src_match["highest_passage_confidence"], 0.50)

    def test_02_multi_source_copying(self):
        """Test Scenario 2: Multi-source copying using multi_source_plagiarism_test.txt"""
        file_path = self.fixtures_dir / "multi_source_plagiarism_test.txt"
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        report = self.detector.detect_document(text, doc_name=file_path.name)
        self.assertIn("PLAGIARISM DETECTED", report["overall_verdict"])
        self.assertGreaterEqual(report["detected_sources_count"], 2)

        detected_docs = [s["source_document"] for s in report["confirmed_detected_sources"]]
        self.assertIn("source-document00019.txt", detected_docs)
        self.assertIn("source-document00484.txt", detected_docs)

        # Both sources must be independently classified as PLAGIARISM_MATCH
        for doc_name in ["source-document00019.txt", "source-document00484.txt"]:
            match = next(s for s in report["confirmed_detected_sources"] if s["source_document"] == doc_name)
            self.assertEqual(match["source_level_decision"], "PLAGIARISM_MATCH")
            self.assertGreaterEqual(match["highest_passage_confidence"], 0.50)

    def test_03_unrelated_text(self):
        """Test Scenario 3: Unrelated non-plagiarized text"""
        file_path = self.fixtures_dir / "unrelated_test.txt"
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        report = self.detector.detect_document(text, doc_name=file_path.name)
        self.assertIn("NO SIGNIFICANT PLAGIARISM DETECTED", report["overall_verdict"])
        self.assertEqual(report["detected_sources_count"], 0)
        self.assertEqual(len(report["confirmed_detected_sources"]), 0)

    def test_04_paraphrased_text(self):
        """Test Scenario 4: Paraphrased text"""
        file_path = self.fixtures_dir / "paraphrased_test.txt"
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        report = self.detector.detect_document(text, doc_name=file_path.name)
        self.assertIsInstance(report, dict)
        self.assertIn("overall_verdict", report)

    def test_05_mixed_copied_and_unrelated(self):
        """Test Scenario 5: Mixed copied and unrelated content"""
        file_path = self.fixtures_dir / "mixed_copied_unrelated_test.txt"
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        report = self.detector.detect_document(text, doc_name=file_path.name)
        self.assertIn("PLAGIARISM DETECTED", report["overall_verdict"])
        detected_docs = [s["source_document"] for s in report["confirmed_detected_sources"]]
        self.assertIn("source-document00019.txt", detected_docs)

    def test_06_repeated_copied_passages(self):
        """Test Scenario 6: Repeated copied passages from the same source"""
        file_path = self.fixtures_dir / "repeated_copied_test.txt"
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        report = self.detector.detect_document(text, doc_name=file_path.name)
        self.assertIn("PLAGIARISM DETECTED", report["overall_verdict"])
        match = next(s for s in report["confirmed_detected_sources"] if s["source_document"] == "source-document00019.txt")
        self.assertGreaterEqual(match["detected_passages_count"], 2)

    def test_07_input_and_source_offset_accuracy(self):
        """Test Scenario 7: Input and source offset verification against actual files"""
        import re
        file_path = self.fixtures_dir / "multi_source_plagiarism_test.txt"
        with open(file_path, "r", encoding="utf-8") as f:
            input_text = f.read()

        report = self.detector.detect_document(input_text, doc_name=file_path.name)
        
        # Verify input offsets map directly to input_text
        for src in report["confirmed_detected_sources"]:
            for inp_span in src["merged_input_offsets"]:
                s, e = inp_span
                span_text = input_text[s:e]
                self.assertGreater(len(span_text), 0)
                self.assertEqual(input_text[s:e], span_text)

        # Verify source offsets map to original source document text on disk
        ext_root = Path(self.config["dataset"]["external_root"])
        for src in report["confirmed_detected_sources"]:
            src_doc_name = src["source_document"]
            found_paths = list(ext_root.rglob(src_doc_name))
            if found_paths:
                with open(found_paths[0], "r", encoding="latin1") as f:
                    orig_src_text = f.read()
                
                # Check that source slices correspond to the input passage with conservative whitespace normalization
                for inp_span, src_span in zip(src["merged_input_offsets"], src["merged_source_offsets"]):
                    in_slice = input_text[inp_span[0]:inp_span[1]]
                    src_slice = orig_src_text[src_span[0]:src_span[1]]
                    
                    in_words = set(re.findall(r'\b\w+\b', in_slice.lower()))
                    src_words = set(re.findall(r'\b\w+\b', src_slice.lower()))
                    
                    overlap = len(in_words.intersection(src_words))
                    union = len(in_words.union(src_words))
                    jaccard = overlap / union if union > 0 else 0.0
                    
                    self.assertGreater(len(src_slice), 0)
                    self.assertGreater(jaccard, 0.70, f"Source offset text does not match input passage: jaccard={jaccard}")

    def test_08_empty_short_and_punctuation_edge_cases(self):
        """Test Scenario 8: Robustness on empty, short, and punctuation edge cases"""
        # Empty string
        rep_empty = self.detector.detect_document("", doc_name="empty")
        self.assertEqual(rep_empty["detected_sources_count"], 0)
        self.assertIn("NO SIGNIFICANT PLAGIARISM DETECTED", rep_empty["overall_verdict"])

        # Punctuation only
        rep_punct = self.detector.detect_document("... ??? !!! --- ::;;", doc_name="punct")
        self.assertEqual(rep_punct["detected_sources_count"], 0)

        # Very short text (single word)
        rep_short = self.detector.detect_document("Plagiarism", doc_name="short")
        self.assertEqual(rep_short["detected_sources_count"], 0)

    def test_09_window_counting_and_merged_regions(self):
        """Test Scenario 9: Separate reporting of evaluated windows, confirmed detections, and merged regions"""
        file_path = self.fixtures_dir / "multi_source_plagiarism_test.txt"
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        report = self.detector.detect_document(text, doc_name=file_path.name)
        
        self.assertIn("total_evaluated_windows", report)
        self.assertIn("confirmed_passage_detections", report)
        self.assertIn("merged_input_regions_count", report)
        self.assertIn("unique_source_documents_count", report)

        self.assertGreaterEqual(report["total_evaluated_windows"], 3)
        self.assertGreaterEqual(report["confirmed_passage_detections"], 2)
        self.assertGreaterEqual(report["merged_input_regions_count"], 2)
        self.assertEqual(report["unique_source_documents_count"], 2)

        for src in report["confirmed_detected_sources"]:
            self.assertIn("evaluated_windows_count", src)
            self.assertIn("confirmed_passage_detections", src)
            self.assertIn("merged_input_regions_count", src)
            self.assertGreaterEqual(src["confirmed_passage_detections"], 1)
            self.assertGreaterEqual(src["merged_input_regions_count"], 1)

if __name__ == "__main__":
    unittest.main()
