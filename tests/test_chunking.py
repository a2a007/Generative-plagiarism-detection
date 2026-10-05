import unittest
from src.preprocessing.chunker import DocumentChunker

class TestDocumentChunker(unittest.TestCase):
    def test_chunk_document_offsets(self):
        text = "Word1 Word2 Word3 Word4 Word5 Word6 Word7 Word8 Word9 Word10"
        chunker = DocumentChunker(words_per_chunk=5, overlap_words=2)
        chunks = chunker.chunk_document("doc1.txt", text)

        self.assertGreater(len(chunks), 0)
        for chk in chunks:
            start = chk["start_char"]
            end = chk["end_char"]
            extracted = text[start:end]
            self.assertEqual(extracted, chk["text"])
            self.assertEqual(chk["doc_id"], "doc1.txt")

if __name__ == "__main__":
    unittest.main()
