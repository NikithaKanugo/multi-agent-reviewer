"""
Shared Memory Module — FAISS Vector Store

This is the SHARED KNOWLEDGE BASE that all agents can query.
It implements the "Retrieval" part of RAG (Retrieval-Augmented Generation).

HOW IT WORKS:
1. Paper chunks are converted to vectors (embeddings) via sentence-transformers (local)
2. Vectors are stored in a FAISS index
3. When an agent needs information, it sends a query
4. The query is embedded and compared against all stored vectors
5. The most similar chunks are returned

SYSTEM DESIGN PARALLEL:
- This is like Redis/Elasticsearch in a microservices architecture
- Multiple services (agents) query it for data
- It's the single source of truth for paper content

WHY A CLASS (not functions)?
- It holds state: the FAISS index + original text chunks
- Multiple agents share ONE instance (same index)
- Clean interface: agents call memory.search() without knowing FAISS internals
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class SharedMemory:
    """FAISS-backed shared memory for multi-agent paper search.

    Usage:
        memory = SharedMemory()
        memory.add_chunks(["chunk1 text", "chunk2 text", ...])
        results = memory.search("What is the methodology?", top_k=3)
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize shared memory.

        Args:
            embedding_model: Sentence-transformers model to use.
                - "all-MiniLM-L6-v2": fast, lightweight, runs locally for free
                - No API key needed — embeddings are computed on your machine
        """
        self.model = SentenceTransformer(embedding_model)
        self.dimension = 384  # all-MiniLM-L6-v2 outputs 384-dim vectors

        # -------------------------------------------------------
        # FAISS INDEX
        # -------------------------------------------------------
        # IndexFlatL2 = brute-force L2 (Euclidean) distance search
        # It compares the query against EVERY stored vector.
        #
        # WHY brute-force (not approximate)?
        # - A paper has ~20-50 chunks. That's tiny.
        # - Approximate search (IndexIVFFlat) shines at 100K+ vectors
        # - For <1000 vectors, brute-force is faster (no training step)
        #
        # INTERVIEW TIP: If asked "how would you scale this?"
        # Answer: "Switch IndexFlatL2 to IndexIVFFlat or IndexHNSW
        # for approximate nearest neighbor search. At millions of
        # vectors, brute-force is too slow."
        # -------------------------------------------------------
        self.index = faiss.IndexFlatL2(self.dimension)

        # Store original text chunks alongside vectors
        # FAISS only stores vectors (numbers), not the original text.
        # We need to map: vector index → original chunk text
        # So when FAISS says "vector #5 is the best match", we can
        # return the actual text of chunk #5.
        self.chunks: list[str] = []

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Convert texts to embedding vectors using a local model.

        This runs entirely on your machine — no API calls, no cost.
        The sentence-transformers library handles tokenization and inference.

        Args:
            texts: List of text strings to embed

        Returns:
            numpy array of shape (len(texts), 384)
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.astype(np.float32)

    def add_chunks(self, chunks: list[str]) -> None:
        """Add text chunks to the FAISS index.

        This is called ONCE during setup, after the paper is loaded and chunked.
        Each chunk gets embedded and stored.

        Args:
            chunks: List of text chunks from paper_loader.chunk_text()

        After this call:
            - self.index contains all chunk vectors
            - self.chunks contains all chunk texts
            - search() is ready to use
        """
        if not chunks:
            raise ValueError("No chunks to add. Is the paper empty?")

        # Store original texts for retrieval later
        self.chunks = chunks

        # Convert all chunks to vectors in ONE API call (batching)
        # WHY batch? One API call with 30 chunks is faster and cheaper
        # than 30 separate API calls with 1 chunk each.
        # This is a production best practice: always batch API calls.
        embeddings = self._embed(chunks)

        # Add vectors to the FAISS index
        # After this, FAISS can search over them
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Search for the most relevant chunks given a query.

        This is what agents call to get information from the paper.

        HOW IT WORKS:
        1. Embed the query (turn question into a vector)
        2. FAISS compares query vector against ALL stored chunk vectors
        3. Returns the top_k closest matches (smallest L2 distance)

        Args:
            query: Natural language question (e.g., "What methodology was used?")
            top_k: Number of results to return (default 3)
                - 1: very focused, might miss context
                - 3: good balance of relevance and coverage
                - 5+: broader but may include less relevant chunks

        Returns:
            List of dicts, each with:
                - "text": the original chunk text
                - "score": L2 distance (LOWER = more similar)
                - "chunk_index": position in the original chunk list
        """
        if self.index.ntotal == 0:
            raise ValueError("Memory is empty. Call add_chunks() first.")

        # Don't try to return more results than we have chunks
        top_k = min(top_k, len(self.chunks))

        # Embed the query (same model as chunks — MUST match!)
        # WHY must it match? Because different models produce different
        # vector spaces. Comparing vectors from different models is like
        # comparing GPS coordinates from different map projections —
        # the numbers mean different things.
        query_vector = self._embed([query])

        # FAISS search: find top_k nearest vectors
        # Returns:
        #   distances: array of L2 distances (lower = more similar)
        #   indices: array of chunk indices (position in self.chunks)
        distances, indices = self.index.search(query_vector, top_k)

        # Build readable results
        results = []
        for i in range(top_k):
            idx = indices[0][i]
            if idx == -1:
                # FAISS returns -1 if there aren't enough results
                continue
            results.append({
                "text": self.chunks[idx],
                "score": float(distances[0][i]),
                "chunk_index": int(idx),
            })

        return results
