import unittest
from scripts.detect_plagiarism import format_compact_terminal_report


class TestTerminalReportFormat(unittest.TestCase):
    def test_compact_report_with_matches(self):
        report = {
            "input_document_name": "multi_source_plagiarism_test.txt",
            "confirmed_detected_sources": [
                {
                    "source_document": "source-document00019.txt",
                    "matching_passage_details": [
                        {
                            "passage_id": "passage_0001",
                            "confidence_score": 0.73,
                            "input_passage_offsets": [399, 950],
                            "source_passage_offsets": [1200, 1751],
                            "input_passage_text": "A" * 200,
                        },
                        {
                            "passage_id": "passage_0002",
                            "confidence_score": 0.81,
                            "input_passage_offsets": [786, 1379],
                            "source_passage_offsets": [1752, 2345],
                            "input_passage_text": "Short passage",
                        },
                    ],
                },
                {
                    "source_document": "source-document00484.txt",
                    "matching_passage_details": [
                        {
                            "passage_id": "passage_0012",
                            "confidence_score": 0.88,
                            "input_passage_offsets": [3879, 4258],
                            "source_passage_offsets": [10972, 11346],
                            "input_passage_text": "Even when I was fully awake, seen for an instant...",
                        }
                    ],
                },
            ],
        }

        output = format_compact_terminal_report(report)

        # Check header
        self.assertIn("============================================================", output)
        self.assertIn("PLAGIARISM DETECTION REPORT", output)
        self.assertIn("Input File       : multi_source_plagiarism_test.txt", output)
        self.assertIn("Overall Verdict  : PLAGIARISM DETECTED", output)
        self.assertIn("Sources Detected : 2", output)
        self.assertIn("Passages Detected: 3", output)

        # Check sources grouping
        self.assertIn("SOURCE 1: source-document00019.txt", output)
        self.assertIn("Passages Detected: 2", output)
        self.assertIn("[1] Passage ID       : passage_0001", output)
        self.assertIn("    Confidence       : 73%", output)
        self.assertIn("    Input Location   : Characters 399-950", output)
        self.assertIn("    Matched Source   : source-document00019.txt", output)
        self.assertIn("    Source Location  : Characters 1200-1751", output)

        # Check truncation to 150 chars + "..."
        self.assertIn("Matched Text     : " + "A" * 150 + "...", output)

        # Check non-truncated text preview
        self.assertIn("Matched Text     : Short passage", output)

        # Check source 2
        self.assertIn("SOURCE 2: source-document00484.txt", output)
        self.assertIn("[1] Passage ID       : passage_0012", output)
        self.assertIn("    Confidence       : 88%", output)
        self.assertIn("    Input Location   : Characters 3879-4258", output)
        self.assertIn("    Source Location  : Characters 10972-11346", output)

        # Ensure no internal/debug feature scores appear
        self.assertNotIn("bm25_score", output.lower())
        self.assertNotIn("dense_similarity", output.lower())
        self.assertNotIn("rrf_score", output.lower())
        self.assertNotIn("jaccard", output.lower())

    def test_missing_source_offsets(self):
        report = {
            "input_document_name": "test_missing_offsets.txt",
            "confirmed_detected_sources": [
                {
                    "source_document": "source-doc.txt",
                    "matching_passage_details": [
                        {
                            "passage_id": "passage_0001",
                            "confidence_score": 0.65,
                            "input_passage_offsets": [10, 50],
                            "source_passage_offsets": None,
                            "input_passage_text": "Sample text",
                        }
                    ],
                }
            ],
        }

        output = format_compact_terminal_report(report)
        self.assertIn("Source Location  : Not available", output)

    def test_no_plagiarism_detected(self):
        report = {
            "input_document_name": "clean_document.txt",
            "confirmed_detected_sources": [],
        }

        output = format_compact_terminal_report(report)
        self.assertIn("Overall Verdict  : NO PLAGIARISM DETECTED", output)
        self.assertIn("Sources Detected : 0", output)
        self.assertIn("Passages Detected: 0", output)
        self.assertNotIn("SOURCE 1", output)


if __name__ == "__main__":
    unittest.main()
