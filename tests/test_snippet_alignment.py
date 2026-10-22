import unittest
from scripts.detect_plagiarism import extract_best_matching_snippet

class TestSnippetAlignment(unittest.TestCase):
    def test_extract_best_matching_snippet_exact_sentence(self):
        query = "This is generally true of both the great arts; but in severity and precision, true of sculpture."
        candidate = (
            "Every accessory in the painting is of value--the fireside, the tiled floor. "
            "113. This is generally true of both the great arts; but in severity and precision, true of sculpture. "
            "To return to our illustration: this poor little girl was more interesting to Edward Frere. "
            "May we sculpture her so? No."
        )

        snippet, start, end = extract_best_matching_snippet(query, candidate, max_chars=150)
        
        self.assertIn("This is generally true of both the great arts", snippet)
        self.assertNotIn("Every accessory in the painting", snippet)
        self.assertGreater(start, 0)
        self.assertLess(end, len(candidate) + 1)

    def test_extract_best_matching_snippet_short_text(self):
        query = "hello world"
        candidate = "hello world"
        snippet, start, end = extract_best_matching_snippet(query, candidate, max_chars=100)
        self.assertEqual(snippet, "hello world")
        self.assertEqual(start, 0)
        self.assertEqual(end, len(candidate))

if __name__ == "__main__":
    unittest.main()
