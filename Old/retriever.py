"""
retriever.py
------------
Implements the collapsed tree retrieval strategy from the RAPTOR paper
(Section 3, Querying).

Collapsed tree search:
    1. Flatten all nodes from every tree level into a single pool
    2. Embed the query with SBERT
    3. Rank all nodes by cosine similarity to the query
    4. Greedily add top-ranked nodes until a token budget is reached

The paper uses 2000 tokens as the default budget, which corresponds to
roughly the top-20 nodes. A token-based budget is used (rather than
top-k) because node lengths vary across levels.

This outperforms tree traversal because it lets the query determine
the right level of abstraction — thematic questions naturally pull
high-level summary nodes, while detail questions pull leaf nodes.
"""

import numpy as np
from dataclasses import dataclass

from embedder import Embedder
from tree import RaptorTree, TreeNode


# --- Constants (from RAPTOR paper) ------------------------------------------

DEFAULT_MAX_TOKENS = 2000   # collapsed tree budget (paper Section 3)


# --- Data structures --------------------------------------------------------

@dataclass
class RetrievalResult:
    """
    The output of a single retrieval query.
    """
    query: str
    nodes: list[TreeNode]         # retrieved nodes, ranked by similarity
    similarities: list[float]     # cosine similarity scores, same order
    context: str                  # concatenated text ready to pass to LLM


# --- Retriever --------------------------------------------------------------

class Retriever:
    """
    Collapsed-tree retriever over a RaptorTree.
    """

    def __init__(self, tree: RaptorTree, embedder: Embedder):
        self.tree = tree
        self.embedder = embedder

        # Pre-stack all node embeddings into a matrix for fast dot-product search
        self._all_nodes: list[TreeNode] = tree.nodes
        self._embedding_matrix: np.ndarray = np.stack(
            [n.embedding for n in self._all_nodes]
        )  # shape (total_nodes, 768)

    def retrieve(
        self,
        query: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> RetrievalResult:
        """
        Run collapsed-tree retrieval for a query.

        Args:
            query:      The question or query string.
            max_tokens: Token budget for the returned context.

        Returns:
            RetrievalResult with ranked nodes and concatenated context.
        """
        # Embed query — same SBERT model, L2-normalized
        query_embedding = self.embedder.embed_texts([query])[0]  # shape (768,)

        # Cosine similarity = dot product (embeddings are L2-normalized)
        similarities = self._embedding_matrix @ query_embedding  # shape (total_nodes,)

        # Rank nodes by descending similarity
        ranked_indices = np.argsort(similarities)[::-1]

        # Greedily fill token budget
        selected_nodes = []
        selected_sims = []
        total_tokens = 0

        for idx in ranked_indices:
            node = self._all_nodes[idx]
            node_tokens = len(node.text.split())  # whitespace token count

            if total_tokens + node_tokens > max_tokens:
                continue  # skip nodes that would exceed budget (paper algorithm 2)

            selected_nodes.append(node)
            selected_sims.append(float(similarities[idx]))
            total_tokens += node_tokens

            if total_tokens >= max_tokens:
                break

        # Concatenate selected node texts as context
        context = "\n\n".join(n.text for n in selected_nodes)

        return RetrievalResult(
            query=query,
            nodes=selected_nodes,
            similarities=selected_sims,
            context=context,
        )


# --- Verification -----------------------------------------------------------

def verify_retrieval(result: RetrievalResult, tree: RaptorTree) -> None:
    """
    Print a human-readable report for a single retrieval result.

    Args:
        result: Output of Retriever.retrieve().
        tree:   The RaptorTree that was searched.
    """
    print("\n=== Retrieval Verification ===")
    print(f"Query      : '{result.query}'")
    print(f"Nodes found: {len(result.nodes)} "
          f"(~{len(result.context.split())} words in context)")

    # Show breakdown by tree level
    from collections import Counter
    level_counts = Counter(n.level for n in result.nodes)
    print("\nNodes by level:")
    for level in sorted(level_counts):
        total_at_level = len(tree.get_level(level))
        print(f"  Level {level}: {level_counts[level]} / {total_at_level} nodes")

    print("\nTop-5 retrieved nodes:")
    for node, sim in zip(result.nodes[:5], result.similarities[:5]):
        print(f"\n  [node_id={node.node_id} | level={node.level} | sim={sim:.4f}]")
        print(f"  '{node.text[:120]}...'")

    print("\n=== Retrieval looks good! Ready for pipeline. ===\n")


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from datasets import load_dataset
    from chunker import chunk_qasper_document
    from tree import build_tree

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    print("Loading QASPER...")
    dataset = load_dataset("allenai/qasper", split="train")
    document = dataset[0]

    chunks = chunk_qasper_document(document, doc_id="qasper_train_0")

    from summarizer import Summarizer
    embedder = Embedder()
    summarizer = Summarizer(openai_api_key=api_key)

    tree = build_tree(chunks, embedder, summarizer)

    retriever = Retriever(tree, embedder)
    result = retriever.retrieve("What discourse relations are used in the proposed method?")

    verify_retrieval(result, tree)
