import unittest
from src.retrieval.candidate_fusion import CandidateFusion

class TestCandidateFusion(unittest.TestCase):
    def test_rrf_scoring(self):
        fusion = CandidateFusion(rrf_k=60.0)

        bm25_results = [
            {"source_document": "sourceA.txt", "score": 20.0, "chunk_id": "c1"},
            {"source_document": "sourceB.txt", "score": 10.0, "chunk_id": "c2"}
        ]

        dense_results = [
            {"source_document": "sourceB.txt", "score": 0.9, "chunk_id": "c3"},
            {"source_document": "sourceA.txt", "score": 0.8, "chunk_id": "c4"}
        ]

        fused = fusion.fuse_results(bm25_results, dense_results, top_k=10)
        self.assertEqual(len(fused), 2)
        
        # sourceB rank: BM25 rank 2 + Dense rank 1 -> 1/(60+2) + 1/(60+1)
        # sourceA rank: BM25 rank 1 + Dense rank 2 -> 1/(60+1) + 1/(60+2)
        # Both appear in top 2
        top_docs = [item["source_document"] for item in fused]
        self.assertIn("sourceA.txt", top_docs)
        self.assertIn("sourceB.txt", top_docs)

if __name__ == "__main__":
    unittest.main()
