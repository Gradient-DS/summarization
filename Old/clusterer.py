"""
clusterer.py
------------
Clusters text embeddings using UMAP + Gaussian Mixture Models (GMM),
exactly as described in the RAPTOR paper.

The pipeline per clustering step is:
    1. Reduce dimensions with UMAP (high-dim embeddings are bad for GMMs)
    2. Fit a GMM for a range of cluster counts
    3. Pick the best cluster count using BIC (penalizes complexity)
    4. Assign chunks to clusters (soft clustering — a chunk can belong to
       multiple clusters if its probability exceeds a threshold)

This is run twice per layer:
    - Global pass: large n_neighbors in UMAP to capture broad themes
    - Local pass:  small n_neighbors within each global cluster to capture
                   fine-grained sub-topics
"""

import numpy as np
from dataclasses import dataclass, field
from sklearn.mixture import GaussianMixture
from sklearn.exceptions import ConvergenceWarning
import umap
import warnings


# --- Constants (from RAPTOR paper) -----------------------------------------

GLOBAL_N_NEIGHBORS = 10   # UMAP neighbors for global clustering pass
LOCAL_N_NEIGHBORS  = 3    # UMAP neighbors for local clustering pass
UMAP_N_COMPONENTS  = 10   # Reduced dimensionality fed into GMM
UMAP_METRIC        = "cosine"
GMM_MAX_CLUSTERS   = 50   # Upper bound on number of clusters to try
GMM_THRESHOLD      = 0.5  # Probability threshold for soft cluster assignment
MIN_CLUSTER_SIZE   = 2    # Clusters smaller than this are dropped
RANDOM_STATE       = 42


# --- Data structures --------------------------------------------------------

@dataclass
class Cluster:
    """
    A group of chunk indices that belong together semantically.
    indices refers to rows in the embedding matrix passed to the clusterer.
    """
    cluster_id: int
    indices: list[int]          # which chunks belong to this cluster
    level: str = "global"       # "global" or "local"
    parent_id: int | None = None  # for local clusters, which global cluster


# --- Core functions ---------------------------------------------------------

def reduce_dimensions(
    embeddings: np.ndarray,
    n_neighbors: int,
    n_components: int = UMAP_N_COMPONENTS,
    metric: str = UMAP_METRIC,
) -> np.ndarray:
    """
    Reduce embedding dimensionality with UMAP before clustering.

    High-dimensional embeddings (768-dim SBERT) cause distance metrics
    to behave poorly in GMMs — everything looks equidistant. UMAP
    projects them into a lower-dimensional space where clusters are
    more separable.

    Args:
        embeddings:    np.ndarray of shape (n, 768).
        n_neighbors:   Controls local vs global structure preservation.
                       Large = global themes, small = local sub-topics.
        n_components:  Target dimensionality (RAPTOR uses 10).
        metric:        Distance metric (cosine works well for text).

    Returns:
        np.ndarray of shape (n, n_components).
    """
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric=metric,
        random_state=RANDOM_STATE,
    )
    return reducer.fit_transform(embeddings)


def find_optimal_clusters(
    reduced_embeddings: np.ndarray,
    max_clusters: int = GMM_MAX_CLUSTERS,
) -> int:
    """
    Fit GMMs for cluster counts from 1 to max_clusters and return
    the count with the lowest BIC score.

    BIC = ln(N)*k - 2*ln(L)
    It rewards goodness-of-fit but penalizes model complexity,
    so it naturally selects the simplest model that explains the data.

    Args:
        reduced_embeddings: UMAP-reduced embeddings, shape (n, n_components).
        max_clusters:       Maximum number of clusters to try.

    Returns:
        Optimal number of clusters (integer >= 1).
    """
    n_samples = len(reduced_embeddings)

    # Can't have more clusters than samples
    max_clusters = min(max_clusters, n_samples - 1)
    if max_clusters < 1:
        return 1

    bic_scores = []
    cluster_range = range(1, max_clusters + 1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for k in cluster_range:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                random_state=RANDOM_STATE,
                max_iter=100,
            )
            gmm.fit(reduced_embeddings)
            bic_scores.append(gmm.bic(reduced_embeddings))

    optimal_k = cluster_range[np.argmin(bic_scores)]
    return optimal_k


def assign_clusters(
    reduced_embeddings: np.ndarray,
    n_clusters: int,
    threshold: float = GMM_THRESHOLD,
) -> np.ndarray:
    """
    Fit a GMM with n_clusters and assign each point to clusters
    where its membership probability exceeds `threshold`.

    Soft clustering: a single chunk can belong to multiple clusters.
    This is important because a paragraph discussing two topics should
    contribute to summaries of both.

    Args:
        reduced_embeddings: UMAP-reduced embeddings, shape (n, n_components).
        n_clusters:         Number of GMM components to fit.
        threshold:          Min probability to assign a point to a cluster.

    Returns:
        np.ndarray of shape (n, n_clusters) with probabilities.
    """
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        random_state=RANDOM_STATE,
        max_iter=100,
    )
    gmm.fit(reduced_embeddings)
    probs = gmm.predict_proba(reduced_embeddings)  # shape (n, n_clusters)
    return probs


def build_clusters_from_probs(
    probs: np.ndarray,
    threshold: float = GMM_THRESHOLD,
    level: str = "global",
    parent_id: int | None = None,
    id_offset: int = 0,
) -> list[Cluster]:
    """
    Convert a probability matrix into a list of Cluster objects.
    Each cluster contains the indices of chunks assigned to it.
    Clusters smaller than MIN_CLUSTER_SIZE are dropped.

    Args:
        probs:      Probability matrix, shape (n, n_clusters).
        threshold:  Min probability for assignment.
        level:      "global" or "local".
        parent_id:  For local clusters, the parent global cluster id.
        id_offset:  Starting cluster id (avoids id collisions across levels).

    Returns:
        List of Cluster objects.
    """
    n_clusters = probs.shape[1]
    clusters = []

    for k in range(n_clusters):
        # All points whose probability of belonging to cluster k >= threshold
        assigned_indices = np.where(probs[:, k] >= threshold)[0].tolist()

        if len(assigned_indices) < MIN_CLUSTER_SIZE:
            continue

        clusters.append(Cluster(
            cluster_id=id_offset + k,
            indices=assigned_indices,
            level=level,
            parent_id=parent_id,
        ))

    return clusters


# --- Main clustering function -----------------------------------------------

def cluster_embeddings(
    embeddings: np.ndarray,
    index_offset: int = 0,
) -> list[Cluster]:
    """
    Full two-pass clustering pipeline as described in the RAPTOR paper.

    Pass 1 (Global): Large n_neighbors in UMAP captures broad themes.
                     Produces global clusters.
    Pass 2 (Local):  Small n_neighbors within each global cluster captures
                     fine-grained sub-topics.
                     Produces local clusters nested inside global ones.

    The final output is the union of global + local clusters.
    Each cluster's indices refer to rows in `embeddings`.
    If index_offset > 0, indices are shifted — used when clustering
    summary nodes at higher tree levels.

    Args:
        embeddings:    np.ndarray of shape (n, 768) — L2-normalized SBERT vectors.
        index_offset:  Shift applied to all cluster indices (for tree building).

    Returns:
        List of Cluster objects (mix of global and local).
    """
    n = len(embeddings)

    if n <= 1:
        # Nothing to cluster
        return [Cluster(cluster_id=0, indices=[0], level="global")]

    print(f"  Clustering {n} embeddings...")

    # ------------------------------------------------------------------
    # Pass 1: Global clustering
    # ------------------------------------------------------------------
    print("  Pass 1: Global clustering (large n_neighbors)...")
    global_n_neighbors = min(GLOBAL_N_NEIGHBORS, n - 1)
    global_reduced = reduce_dimensions(
        embeddings,
        n_neighbors=global_n_neighbors,
        n_components=min(UMAP_N_COMPONENTS, n - 2),  # spectral init needs k+1 < N, so n_components <= n-2
    )

    n_global = find_optimal_clusters(global_reduced)
    print(f"    Optimal global clusters (BIC): {n_global}")

    global_probs = assign_clusters(global_reduced, n_clusters=n_global)
    global_clusters = build_clusters_from_probs(
        global_probs,
        level="global",
        id_offset=0,
    )
    print(f"    Global clusters formed: {len(global_clusters)}")

    # ------------------------------------------------------------------
    # Pass 2: Local clustering within each global cluster
    # ------------------------------------------------------------------
    print("  Pass 2: Local clustering within each global cluster...")
    local_clusters = []
    local_id_offset = len(global_clusters)

    for g_cluster in global_clusters:
        member_indices = g_cluster.indices
        if len(member_indices) < 3:  # UMAP requires n_neighbors >= 2, so need >= 3 points
            continue

        member_embeddings = embeddings[member_indices]
        local_n_neighbors = min(LOCAL_N_NEIGHBORS, len(member_indices) - 1)

        local_reduced = reduce_dimensions(
            member_embeddings,
            n_neighbors=local_n_neighbors,
            n_components=min(UMAP_N_COMPONENTS, len(member_indices) - 2),  # spectral init needs k+1 < N
        )

        n_local = find_optimal_clusters(local_reduced)
        local_probs = assign_clusters(local_reduced, n_clusters=n_local)

        # Map local indices back to global indices
        sub_clusters = build_clusters_from_probs(
            local_probs,
            level="local",
            parent_id=g_cluster.cluster_id,
            id_offset=local_id_offset,
        )

        # Remap indices: local pass operates on member_embeddings,
        # but we need indices into the original embeddings array
        for sub in sub_clusters:
            sub.indices = [member_indices[i] for i in sub.indices]

        local_clusters.extend(sub_clusters)
        local_id_offset += len(sub_clusters)

    print(f"    Local clusters formed: {len(local_clusters)}")

    # ------------------------------------------------------------------
    # Combine and apply index_offset (for higher tree levels)
    # ------------------------------------------------------------------
    all_clusters = global_clusters + local_clusters

    if index_offset > 0:
        for cluster in all_clusters:
            cluster.indices = [i + index_offset for i in cluster.indices]

    print(f"  Total clusters: {len(all_clusters)}")
    return all_clusters


# --- Verification -----------------------------------------------------------

def verify_clusters(
    clusters: list[Cluster],
    embeddings: np.ndarray,
    chunks=None,
) -> None:
    """
    Print a human-readable report of clustering results.
    Optionally shows chunk text if chunks are passed in.

    Args:
        clusters:   Output of cluster_embeddings().
        embeddings: Original embedding matrix.
        chunks:     Optional list of Chunk objects for text preview.
    """
    print("\n=== Cluster Verification ===")
    print(f"Total clusters: {len(clusters)}")

    global_clusters = [c for c in clusters if c.level == "global"]
    local_clusters  = [c for c in clusters if c.level == "local"]
    print(f"  Global: {len(global_clusters)}  |  Local: {len(local_clusters)}")

    sizes = [len(c.indices) for c in clusters]
    print(f"  Cluster sizes — min: {min(sizes)}, "
          f"max: {max(sizes)}, avg: {np.mean(sizes):.1f}")

    # Soft clustering stats: how many clusters does each chunk appear in?
    from collections import Counter
    chunk_cluster_count = Counter()
    for c in clusters:
        for idx in c.indices:
            chunk_cluster_count[idx] += 1
    multi_assigned = sum(1 for v in chunk_cluster_count.values() if v > 1)
    print(f"\n  Chunks assigned to >1 cluster (soft): "
          f"{multi_assigned} / {len(embeddings)}")

    # Show a sample of 3 clusters with their chunk previews
    print("\n  Sample clusters:")
    for cluster in clusters[:3]:
        print(f"\n  Cluster {cluster.cluster_id} "
              f"[{cluster.level}] — {len(cluster.indices)} members")
        for idx in cluster.indices[:3]:
            if chunks:
                text_preview = chunks[idx].text[:80]
                section = chunks[idx].section
                print(f"    [{idx}] ({section}) {text_preview}...")
            else:
                print(f"    [{idx}] embedding norm: "
                      f"{np.linalg.norm(embeddings[idx]):.4f}")

    print("\n=== Clustering looks good! Ready for summarizer. ===\n")


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from datasets import load_dataset
    from chunker import chunk_qasper_document
    from embedder import Embedder

    print("Loading QASPER...")
    dataset = load_dataset("allenai/qasper", split="train")
    document = dataset[0]

    chunks = chunk_qasper_document(document, doc_id="qasper_train_0")
    print(f"Chunks: {len(chunks)}")

    embedder = Embedder()
    embeddings = embedder.embed_chunks(chunks)
    print(f"Embeddings: {embeddings.shape}")

    clusters = cluster_embeddings(embeddings)
    verify_clusters(clusters, embeddings, chunks)
