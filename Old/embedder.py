"""
embedder.py
-----------
Embeds text chunks using SBERT (multi-qa-mpnet-base-cos-v1),
the exact model used in the RAPTOR paper.

Each chunk is embedded into a 768-dimensional dense vector.
These vectors are used for clustering (GMM + UMAP) and
for cosine similarity search during retrieval.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from chunker import Chunk


# RAPTOR paper uses this exact model
SBERT_MODEL_NAME = "multi-qa-mpnet-base-cos-v1"


class Embedder:
    """
    Wraps a SentenceTransformer model and embeds Chunk objects.
    The model is loaded once and reused across all calls.
    """

    def __init__(self, model_name: str = SBERT_MODEL_NAME):
        print(f"Loading SBERT model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.embedding_dim}")

    def embed_chunks(self, chunks: list[Chunk], batch_size: int = 64) -> np.ndarray:
        """
        Embed a list of Chunk objects.

        Args:
            chunks:     List of Chunk objects from chunker.py.
            batch_size: Number of chunks to embed at once. Reduce if OOM.

        Returns:
            np.ndarray of shape (len(chunks), embedding_dim).
            Row i corresponds to chunks[i].
        """
        texts = [chunk.text for chunk in chunks]
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True  # cosine similarity = dot product after L2 norm
        )
        return embeddings

    def embed_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """
        Embed raw strings directly — used for query embedding at retrieval time
        and for re-embedding cluster summaries during tree construction.

        Args:
            texts:      List of raw strings to embed.
            batch_size: Number of texts to embed at once.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim).
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings


def verify_embeddings(embeddings: np.ndarray, chunks: list[Chunk]) -> None:
    """
    Run a set of sanity checks on the embedding matrix and print a report.
    Helps catch issues early before passing embeddings to the clusterer.

    Checks:
    - Shape is correct
    - No NaN or Inf values
    - Vectors are L2-normalized (norms close to 1.0)
    - Semantically similar chunks have higher cosine similarity than random pairs
    """
    print("\n=== Embedding Verification ===")

    # 1. Shape
    print(f"\n[1] Shape: {embeddings.shape}")
    assert embeddings.shape[0] == len(chunks), "Row count doesn't match chunk count!"
    assert embeddings.shape[1] == 768, f"Expected 768 dims, got {embeddings.shape[1]}"
    print("    ✓ Shape is correct")

    # 2. No NaN or Inf
    assert not np.isnan(embeddings).any(), "NaN values found in embeddings!"
    assert not np.isinf(embeddings).any(), "Inf values found in embeddings!"
    print("[2] ✓ No NaN or Inf values")

    # 3. L2 norms — should be ~1.0 since we normalize
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"[3] L2 norms — mean: {norms.mean():.4f}, "
          f"min: {norms.min():.4f}, max: {norms.max():.4f}")
    assert np.allclose(norms, 1.0, atol=1e-4), "Embeddings are not L2-normalized!"
    print("    ✓ All vectors are unit-normalized")

    # 4. Cosine similarity spot check
    # Since vectors are normalized, cosine similarity = dot product
    print("\n[4] Cosine similarity spot check:")

    # Adjacent chunks (same section) should be more similar than random pairs
    if len(chunks) >= 4:
        sim_adjacent = float(embeddings[0] @ embeddings[1])
        sim_random   = float(embeddings[0] @ embeddings[-1])
        print(f"    Chunk 0 vs Chunk 1 (adjacent):  {sim_adjacent:.4f}")
        print(f"    Chunk 0 vs Chunk -1 (distant):  {sim_random:.4f}")

    # Show the top-3 most similar pairs to chunk 0
    sims_to_first = embeddings @ embeddings[0]
    top3_indices = np.argsort(sims_to_first)[::-1][1:4]  # skip self (index 0)
    print(f"\n    Top-3 most similar chunks to chunk 0:")
    print(f"    (section: '{chunks[0].section}' | '{chunks[0].text[:60]}...')")
    for idx in top3_indices:
        print(f"      → Chunk {idx} (sim={sims_to_first[idx]:.4f}) "
              f"section='{chunks[idx].section}' | "
              f"'{chunks[idx].text[:60]}...'")

    print("\n=== All checks passed! Ready for clustering. ===\n")


# ---------------------------------------------------------------------------
# Notebook-friendly usage example (also works as a standalone script)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from datasets import load_dataset
    from chunker import chunk_qasper_document

    # Load one document
    print("Loading QASPER dataset...")
    dataset = load_dataset("allenai/qasper", split="train")
    document = dataset[0]

    # Chunk it
    chunks = chunk_qasper_document(document, doc_id="qasper_train_0")
    print(f"Chunked into {len(chunks)} chunks")

    # Embed
    embedder = Embedder()
    embeddings = embedder.embed_chunks(chunks)

    # Verify
    verify_embeddings(embeddings, chunks)