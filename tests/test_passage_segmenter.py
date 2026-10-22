import unittest
from src.preprocessing.passage_segmenter import PassageSegmenter

class TestPassageSegmenter(unittest.TestCase):
    def test_exact_character_offsets(self):
        text = "This is a sample document for testing character offsets. " * 30
        segmenter = PassageSegmenter(window_words=20, overlap_words=5, min_words=10)
        passages = segmenter.segment_text(text)
        
        self.assertGreater(len(passages), 1)
        for p in passages:
            extracted = text[p["start_char"]:p["end_char"]]
            self.assertEqual(extracted, p["text"], f"Offset mismatch in passage {p['passage_id']}")
            self.assertEqual(p["word_count"], len(p["text"].split()))

    def test_short_document(self):
        text = "A short document containing only seven words."
        segmenter = PassageSegmenter(window_words=50, overlap_words=10)
        passages = segmenter.segment_text(text)
        
        self.assertEqual(len(passages), 1)
        self.assertEqual(passages[0]["passage_id"], "passage_0001")
        self.assertEqual(passages[0]["text"], text)
        self.assertEqual(passages[0]["start_char"], 0)
        self.assertEqual(passages[0]["end_char"], len(text))
        self.assertEqual(passages[0]["word_count"], 7)

    def test_empty_and_whitespace_text(self):
        segmenter = PassageSegmenter(window_words=20, overlap_words=5)
        self.assertEqual(segmenter.segment_text(""), [])
        self.assertEqual(segmenter.segment_text("   \n\t  "), [])

    def test_punctuation_and_newlines(self):
        text = "Hello, world! -- Here's a test: [part 1]... \n\nNew paragraph with (brackets) and 'quotes'."
        segmenter = PassageSegmenter(window_words=5, overlap_words=2, min_words=2)
        passages = segmenter.segment_text(text)
        
        self.assertGreater(len(passages), 0)
        for p in passages:
            self.assertEqual(text[p["start_char"]:p["end_char"]], p["text"])

    def test_window_overlap(self):
        words = [f"word_{i}" for i in range(100)]
        text = " ".join(words)
        segmenter = PassageSegmenter(window_words=30, overlap_words=10, min_words=5, split_paragraphs=False)
        passages = segmenter.segment_text(text)
        
        # In a step of 20 (30 - 10), passage 1 has 0..30, passage 2 has 20..50
        p1_words = passages[0]["text"].split()
        p2_words = passages[1]["text"].split()
        overlap = set(p1_words).intersection(set(p2_words))
        self.assertEqual(len(overlap), 10)

    def test_paragraph_aware_segmentation(self):
        para1 = "This is the first independent paragraph with exactly nine words."
        para2 = "And this is the second independent paragraph with eight words."
        full_doc = para1 + "\n\n" + para2
        segmenter = PassageSegmenter(window_words=100, overlap_words=30, min_words=5, split_paragraphs=True)
        passages = segmenter.segment_text(full_doc)

        self.assertEqual(len(passages), 2)
        self.assertEqual(passages[0]["text"], para1)
        self.assertEqual(passages[1]["text"], para2)
        self.assertEqual(full_doc[passages[0]["start_char"]:passages[0]["end_char"]], para1)
        self.assertEqual(full_doc[passages[1]["start_char"]:passages[1]["end_char"]], para2)

    def test_crlf_paragraph_aware_segmentation(self):
        para1 = "This is the first CRLF paragraph with some words."
        para2 = "And this is the second CRLF paragraph with different words."
        full_doc = para1 + "\r\n\r\n" + para2
        segmenter = PassageSegmenter(window_words=100, overlap_words=30, min_words=5, split_paragraphs=True)
        passages = segmenter.segment_text(full_doc)

        self.assertEqual(len(passages), 2)
        self.assertEqual(passages[0]["text"], para1)
        self.assertEqual(passages[1]["text"], para2)
        self.assertEqual(full_doc[passages[0]["start_char"]:passages[0]["end_char"]], para1)
        self.assertEqual(full_doc[passages[1]["start_char"]:passages[1]["end_char"]], para2)

if __name__ == "__main__":
    unittest.main()
