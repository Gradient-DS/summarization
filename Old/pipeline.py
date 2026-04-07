"""
pipeline.py
-----------
RAPTOR summarization subagent. Called by the general agent when a summary
request is detected — intent detection happens upstream, not here.

Receives a RaptorTree (already built) and returns a summary of the document
using the root-level nodes, which are the most compressed representation.
"""

from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from embedder import Embedder
from tree import RaptorTree
from retriever import Retriever, RetrievalResult, DEFAULT_MAX_TOKENS


# --- Summary prompt ---------------------------------------------------------

SUMMARY_SYSTEM = "You are a document summarization assistant."
SUMMARY_USER = (
    "Write a concise but complete summary of the document based on the "
    "following content:\n\n{context}"
)


# --- Data structure ---------------------------------------------------------

@dataclass
class PipelineResult:
    query: str
    retrieval: RetrievalResult     # nodes + context that were used
    answer: str                    # final summary


# --- Pipeline ---------------------------------------------------------------

class Pipeline:
    """
    RAPTOR summarization subagent.
    Caller is responsible for routing — only pass summary requests here.
    """

    def __init__(
        self,
        tree: RaptorTree,
        embedder: Embedder,
        model: str = "gpt-4.1-nano",
        openai_api_key: str | None = None,
    ):
        self.tree = tree
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.0,
            openai_api_key=openai_api_key,
        )

    def _get_summary_context(self) -> str:
        """Use the highest-level (root) nodes as context — most compressed view."""
        top_level = self.tree.num_levels - 1
        nodes = self.tree.get_level(top_level)
        if not nodes:
            nodes = self.tree.nodes
        return "\n\n".join(n.text for n in nodes)

    def run(self, query: str = "Summarize this document.") -> PipelineResult:
        """
        Generate a summary of the document.

        Args:
            query: Optional query string — passed through for logging purposes.
                   Intent detection is handled upstream by the general agent.

        Returns:
            PipelineResult with the root-level nodes used and the summary.
        """
        context = self._get_summary_context()
        root_nodes = self.tree.get_level(self.tree.num_levels - 1)

        retrieval = RetrievalResult(
            query=query,
            nodes=root_nodes,
            similarities=[1.0] * len(root_nodes),
            context=context,
        )

        answer = self.llm.invoke([
            SystemMessage(content=SUMMARY_SYSTEM),
            HumanMessage(content=SUMMARY_USER.format(context=context)),
        ]).content.strip()

        return PipelineResult(query=query, retrieval=retrieval, answer=answer)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from datasets import load_dataset
    from chunker import chunk_qasper_document
    from embedder import Embedder
    from summarizer import Summarizer
    from tree import build_tree

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    dataset = load_dataset("allenai/qasper", split="train")
    document = dataset[0]
    chunks = chunk_qasper_document(document, doc_id="qasper_train_0")

    embedder = Embedder()
    summarizer = Summarizer(openai_api_key=api_key)
    tree = build_tree(chunks, embedder, summarizer)

    pipeline = Pipeline(tree, embedder, openai_api_key=api_key)
    result = pipeline.run()
    print(result.answer)
