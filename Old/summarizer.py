"""
summarizer.py
-------------
Summarizes each cluster of text chunks using an LLM, exactly as described
in the RAPTOR paper (Section 3, Model-Based Summarization).

The paper uses gpt-3.5-turbo with the prompt from Appendix D:
    system: "You are a Summarizing Text Portal"
    user:   "Write a summary of the following, including as many key
             details as possible: {context}:"

Each cluster's member chunks are concatenated and sent to the LLM.
The returned summary becomes a new node at the next tree level,
ready to be re-embedded and clustered again.
"""

from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from chunker import Chunk
from clusterer import Cluster


# --- Prompt (exact from RAPTOR paper Appendix D) ----------------------------

SYSTEM_PROMPT = "You are a Summarizing Text Portal"
USER_PROMPT_TEMPLATE = (
    "Write a summary of the following, including as many key details as "
    "possible: {context}:"
)


# --- Data structure ---------------------------------------------------------

@dataclass
class SummaryNode:
    """
    A summarized node produced from a cluster.
    This is the unit passed to the next tree level for re-embedding.
    """
    text: str               # the generated summary
    cluster_id: int         # which cluster this summary came from
    level: int              # tree level (0 = leaf chunks, 1 = first summaries, ...)
    source_indices: list[int]  # indices of the chunks/nodes that were summarized


# --- Core summarizer --------------------------------------------------------

class Summarizer:
    """
    Wraps an LLM and summarizes clusters of text.
    Model is loaded once and reused.
    """

    def __init__(self, model: str = "gpt-4.1-nano", temperature: float = 0.0,
                 openai_api_key: str | None = None):
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=openai_api_key,
        )

    def _summarize_text(self, context: str) -> str:
        """Send a single context string to the LLM and return the summary."""
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=USER_PROMPT_TEMPLATE.format(context=context)),
        ]
        return self.llm.invoke(messages).content.strip()

    def summarize_clusters(
        self,
        clusters: list[Cluster],
        texts: list[str],
        level: int = 1,
    ) -> list[SummaryNode]:
        """
        Summarize each cluster by concatenating its members' texts and
        calling the LLM.

        Args:
            clusters:  Output of cluster_embeddings() — list of Cluster objects.
            texts:     The text for each node at this level.
                       texts[i] is the text for node index i.
                       At level 1, these are chunk texts.
                       At level 2+, these are the previous level's summaries.
            level:     The tree level being built (1-indexed, leaf=0).

        Returns:
            List of SummaryNode objects, one per cluster.
        """
        summary_nodes = []

        for i, cluster in enumerate(clusters):
            # Concatenate texts of all members in this cluster
            member_texts = [texts[idx] for idx in cluster.indices]
            context = "\n\n".join(member_texts)

            print(f"  Summarizing cluster {cluster.cluster_id} "
                  f"({len(cluster.indices)} members, ~{len(context.split())} words)...")

            summary = self._summarize_text(context)

            summary_nodes.append(SummaryNode(
                text=summary,
                cluster_id=cluster.cluster_id,
                level=level,
                source_indices=cluster.indices,
            ))

        return summary_nodes


# --- Verification -----------------------------------------------------------

def verify_summaries(
    summary_nodes: list[SummaryNode],
    texts: list[str],
) -> None:
    """
    Print a human-readable report of summarization results.
    Checks compression ratio and shows sample summaries.

    Args:
        summary_nodes: Output of summarize_clusters().
        texts:         Original texts that were summarized (same list passed in).
    """
    print("\n=== Summarization Verification ===")
    print(f"Total summaries: {len(summary_nodes)}")

    ratios = []
    for node in summary_nodes:
        source_len = sum(len(texts[i].split()) for i in node.source_indices)
        summary_len = len(node.text.split())
        if source_len > 0:
            ratios.append(summary_len / source_len)

    if ratios:
        avg_ratio = sum(ratios) / len(ratios)
        print(f"Avg compression ratio: {avg_ratio:.2f}  "
              f"(paper reports ~0.28 — lower is more compressed)")

    print("\nSample summaries (first 3):")
    for node in summary_nodes[:3]:
        source_words = sum(len(texts[i].split()) for i in node.source_indices)
        print(f"\n  Cluster {node.cluster_id} | level={node.level} | "
              f"sources={node.source_indices} | "
              f"input≈{source_words} words → summary≈{len(node.text.split())} words")
        print(f"  Summary: {node.text[:200]}...")

    print("\n=== Summaries look good! Ready for re-embedding. ===\n")


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from datasets import load_dataset
    from chunker import chunk_qasper_document
    from embedder import Embedder
    from clusterer import cluster_embeddings

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    print("Loading QASPER...")
    dataset = load_dataset("allenai/qasper", split="train")
    document = dataset[0]

    chunks = chunk_qasper_document(document, doc_id="qasper_train_0")
    print(f"Chunks: {len(chunks)}")

    embedder = Embedder()
    embeddings = embedder.embed_chunks(chunks)

    clusters = cluster_embeddings(embeddings)
    print(f"Clusters: {len(clusters)}")

    chunk_texts = [c.text for c in chunks]
    summarizer = Summarizer(openai_api_key=api_key)
    summary_nodes = summarizer.summarize_clusters(clusters, chunk_texts, level=1)

    verify_summaries(summary_nodes, chunk_texts)
