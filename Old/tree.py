"""
tree.py
-------
Builds the RAPTOR tree by recursively embedding, clustering, and summarizing
chunks of text, as described in Section 3 of the RAPTOR paper.

Tree construction loop (per level):
    1. Embed the current nodes (chunks at level 0, summaries at level 1+)
    2. Cluster the embeddings (GMM + UMAP, two-pass global/local)
    3. Summarize each cluster → produces new nodes one level up
    4. Repeat until further clustering is not possible (≤1 cluster)

All nodes (leaf chunks + all summary levels) are stored flat in the tree,
so the retriever can do collapsed-tree search across every level at once.
"""

import numpy as np
from dataclasses import dataclass, field

from chunker import Chunk
from embedder import Embedder
from clusterer import cluster_embeddings, MIN_CLUSTER_SIZE
from summarizer import Summarizer, SummaryNode


# --- Data structures --------------------------------------------------------

@dataclass
class TreeNode:
    """
    A single node in the RAPTOR tree.
    Level 0 = leaf chunk.  Level 1+ = LLM-generated summary.
    """
    node_id: int
    text: str
    embedding: np.ndarray        # shape (768,) — set after embed step
    level: int                   # 0 = leaf, 1 = first summary layer, ...
    children: list[int] = field(default_factory=list)  # node_ids of source nodes


@dataclass
class RaptorTree:
    """
    Stores every node in the tree, indexed by node_id.
    nodes[i] is the TreeNode with node_id == i.
    """
    nodes: list[TreeNode] = field(default_factory=list)

    def get_level(self, level: int) -> list[TreeNode]:
        return [n for n in self.nodes if n.level == level]

    @property
    def num_levels(self) -> int:
        if not self.nodes:
            return 0
        return max(n.level for n in self.nodes) + 1


# --- Tree builder -----------------------------------------------------------

def build_tree(
    chunks: list[Chunk],
    embedder: Embedder,
    summarizer: Summarizer,
    max_levels: int = 10,
) -> RaptorTree:
    """
    Build the full RAPTOR tree bottom-up from leaf chunks.

    Args:
        chunks:      Leaf-level text chunks from chunker.py.
        embedder:    Embedder instance (SBERT).
        summarizer:  Summarizer instance (LLM).
        max_levels:  Safety cap on recursion depth.

    Returns:
        RaptorTree containing all nodes across all levels.
    """
    tree = RaptorTree()
    node_id_counter = 0

    # ------------------------------------------------------------------
    # Level 0: embed and store the leaf chunks
    # ------------------------------------------------------------------
    print("=== Building RAPTOR Tree ===")
    print(f"\n[Level 0] Embedding {len(chunks)} leaf chunks...")

    leaf_embeddings = embedder.embed_chunks(chunks)

    for i, chunk in enumerate(chunks):
        tree.nodes.append(TreeNode(
            node_id=node_id_counter,
            text=chunk.text,
            embedding=leaf_embeddings[i],
            level=0,
            children=[],
        ))
        node_id_counter += 1

    print(f"  Stored {len(chunks)} leaf nodes.")

    # ------------------------------------------------------------------
    # Recursive levels: cluster → summarize → re-embed → repeat
    # ------------------------------------------------------------------
    current_level = 0

    for _ in range(max_levels):
        current_nodes = tree.get_level(current_level)

        if len(current_nodes) <= MIN_CLUSTER_SIZE:
            print(f"\n[Level {current_level}] Only {len(current_nodes)} nodes — "
                  "stopping (cannot cluster further).")
            break

        print(f"\n[Level {current_level} → {current_level + 1}] "
              f"Clustering {len(current_nodes)} nodes...")

        # Stack embeddings for clustering
        current_embeddings = np.stack([n.embedding for n in current_nodes])
        current_ids = [n.node_id for n in current_nodes]

        clusters = cluster_embeddings(current_embeddings)

        if len(clusters) <= 1:
            print(f"  Only {len(clusters)} cluster formed — stopping.")
            break

        # Summarize each cluster
        print(f"  Summarizing {len(clusters)} clusters with LLM...")
        current_texts = [n.text for n in current_nodes]
        summary_nodes = summarizer.summarize_clusters(
            clusters, current_texts, level=current_level + 1
        )

        # Embed the summaries
        summary_texts = [s.text for s in summary_nodes]
        print(f"  Re-embedding {len(summary_texts)} summaries...")
        summary_embeddings = embedder.embed_texts(summary_texts)

        # Store summaries as new TreeNodes
        new_level = current_level + 1
        for j, (snode, emb) in enumerate(zip(summary_nodes, summary_embeddings)):
            # Map local cluster indices → global node_ids
            child_ids = [current_ids[i] for i in snode.source_indices]

            tree.nodes.append(TreeNode(
                node_id=node_id_counter,
                text=snode.text,
                embedding=emb,
                level=new_level,
                children=child_ids,
            ))
            node_id_counter += 1

        print(f"  Added {len(summary_nodes)} summary nodes at level {new_level}.")
        current_level = new_level

    print(f"\n=== Tree complete: {len(tree.nodes)} total nodes "
          f"across {tree.num_levels} levels ===\n")
    return tree


# --- Verification -----------------------------------------------------------

def verify_tree(tree: RaptorTree) -> None:
    """Print a summary of the tree structure."""
    print("\n=== Tree Verification ===")
    print(f"Total nodes : {len(tree.nodes)}")
    print(f"Total levels: {tree.num_levels}")

    for level in range(tree.num_levels):
        nodes = tree.get_level(level)
        avg_words = np.mean([len(n.text.split()) for n in nodes])
        print(f"  Level {level}: {len(nodes):3d} nodes | "
              f"avg {avg_words:.0f} words/node")

    print("\nSample node at each level:")
    for level in range(tree.num_levels):
        node = tree.get_level(level)[0]
        print(f"\n  Level {level} | node_id={node.node_id} | "
              f"children={node.children[:4]}{'...' if len(node.children) > 4 else ''}")
        print(f"  '{node.text[:120]}...'")

    print("\n=== Tree looks good! Ready for retriever. ===\n")


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from datasets import load_dataset
    from chunker import chunk_qasper_document

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    print("Loading QASPER...")
    dataset = load_dataset("allenai/qasper", split="train")
    document = dataset[0]

    chunks = chunk_qasper_document(document, doc_id="qasper_train_0")
    print(f"Chunks: {len(chunks)}")

    embedder = Embedder()
    summarizer = Summarizer(openai_api_key=api_key)

    tree = build_tree(chunks, embedder, summarizer)
    verify_tree(tree)
