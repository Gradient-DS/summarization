"""
booksum_pipeline.py
-------------------
BookSum-specific helpers that sit on top of the shared RAPTOR pipeline.

BookSum dataset structure (kmfoda/booksum):
    {
      "book_id":      str,   e.g. "The Last of the Mohicans.chapter 3"
      "summary_id":   str,   e.g. "chapter 3"
      "chapter":      str,   full chapter text  ← fed to RAPTOR
      "summary_text": str,   human-written summary  ← reference for evaluation
      "is_aggregate": bool,  True = multi-chapter roll-up (we skip these)
      "chapter_length": float,
      "summary_length": float,
    }
"""

import os
import re
import sys

_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_root, "lib"))   # for `from raptor import ...`
sys.path.insert(0, os.path.dirname(__file__))    # for `from prompts/qasper_pipeline import ...`

from qasper_pipeline import (
    RetrievalAugmentationConfig,
    SummaryResult,
    EvalScores,
    build_raptor_config,
    chain_of_density_summarize,
    evaluate_summary,
    print_evaluation,
    compare,
)
from openai import OpenAI
from prompts import DOCUMENT_SUMMARY_SYSTEM, DOCUMENT_SUMMARY_USER, RETRIEVAL_QUERY
from raptor import RetrievalAugmentation

DEFAULT_CACHE_DIR = os.path.join(_root, ".tree_cache")


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(title: str, cache_dir: str) -> str:
    """Return the file path for a cached tree, derived from the chapter title."""
    safe = re.sub(r"[^\w\-]", "_", title)[:80]
    return os.path.join(cache_dir, f"{safe}.pkl")


def build_tree(
    row: dict,
    config: RetrievalAugmentationConfig = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
    force_rebuild: bool = False,
) -> RetrievalAugmentation:
    """
    Build (or load from cache) a RAPTOR tree for a BookSum chapter.

    The tree is saved to `cache_dir/<chapter_title>.pkl` after building.
    On subsequent calls the cached tree is loaded instead of rebuilding,
    saving both time and API cost.

    Args:
        row:           A single BookSum dataset row.
        config:        Optional pre-built RetrievalAugmentationConfig.
        cache_dir:     Directory to store cached trees. Created if absent.
        force_rebuild: If True, ignore the cache and rebuild from scratch.

    Returns:
        A RetrievalAugmentation instance ready for retrieval.
    """
    text, _, title = extract_booksum_text(row)

    if config is None:
        config = build_raptor_config()

    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_path(title, cache_dir)

    if not force_rebuild and os.path.exists(path):
        print(f"Loading cached tree: {path}")
        return RetrievalAugmentation(config=config, tree=path)

    print(f"Building tree for: {title}")
    ra = RetrievalAugmentation(config=config)
    ra.add_documents(text)
    ra.save(path)
    print(f"Tree saved to: {path}")
    return ra


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_booksum(split: str = "validation", single_chapters_only: bool = True):
    """
    Load the BookSum dataset, optionally filtered to single chapters only.

    Args:
        split:                Dataset split ("train", "validation", "test").
        single_chapters_only: If True, filters out aggregate (multi-chapter) rows
                              and rows with empty chapter text.

    Returns:
        HuggingFace Dataset object.
    """
    import warnings
    import logging
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)

    from datasets import load_dataset
    ds = load_dataset("kmfoda/booksum", split=split)

    logging.disable(logging.NOTSET)

    if single_chapters_only:
        ds = ds.filter(lambda row: not row["is_aggregate"] and bool(row["chapter"] and row["chapter"].strip()))

    return ds


def extract_booksum_text(row: dict) -> tuple[str, str, str]:
    """
    Extract text, reference summary, and title from a BookSum row.

    Returns:
        (text, reference, title)
    """
    text = row.get("chapter", "") or ""
    reference = row.get("summary_text", "") or ""
    title = row.get("book_id", "") or ""
    return text.strip(), reference.strip(), title


# ---------------------------------------------------------------------------
# Summarize a chapter
# ---------------------------------------------------------------------------

def summarize_chapter(
    row: dict,
    config: RetrievalAugmentationConfig = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
    force_rebuild: bool = False,
) -> SummaryResult:
    """
    Build (or load) a RAPTOR tree for a BookSum chapter and generate a summary.

    Args:
        row:           A single BookSum dataset row.
        config:        Optional pre-built RetrievalAugmentationConfig.
        cache_dir:     Directory for cached trees.
        force_rebuild: Ignore cache and rebuild the tree.

    Returns:
        SummaryResult with title, abstract (=summary_text), summary, scores={}.
    """
    _, reference, title = extract_booksum_text(row)

    ra = build_tree(row, config=config, cache_dir=cache_dir, force_rebuild=force_rebuild)

    context, _ = ra.retrieve(
        question=RETRIEVAL_QUERY,
        top_k=20,
        max_tokens=10000,
        collapse_tree=True,
        return_layer_information=True,
    )

    client = OpenAI()
    summary = chain_of_density_summarize(context, client)

    return SummaryResult(title=title, abstract=reference, summary=summary, scores={})


# ---------------------------------------------------------------------------
# Query a chapter (the "What does this book say about X?" use case)
# ---------------------------------------------------------------------------

def query_chapter(
    row: dict,
    question: str,
    config: RetrievalAugmentationConfig = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
    force_rebuild: bool = False,
) -> str:
    """
    Answer a question about a BookSum chapter using RAPTOR tree retrieval.
    Reuses a cached tree if available — no rebuild needed per question.

    Example:
        query_chapter(row, "What does this chapter say about friendship?")
    """
    ra = build_tree(row, config=config, cache_dir=cache_dir, force_rebuild=force_rebuild)

    context, _ = ra.retrieve(
        question=question,
        top_k=20,
        max_tokens=10000,
        collapse_tree=True,
        return_layer_information=True,
    )

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": "You are a literary assistant."},
            {
                "role": "user",
                "content": (
                    f"Based on the following text, answer this question: {question}\n\n"
                    f"Text:\n{context}"
                ),
            },
        ],
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def compare_chapter(
    row: dict,
    config: RetrievalAugmentationConfig = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
    force_rebuild: bool = False,
    use_judge: bool = True,
    judge_model: str = "gpt-4.1",
) -> dict:
    """Convenience wrapper: run RAPTOR vs baseline comparison on a BookSum row."""
    text, reference, title = extract_booksum_text(row)

    # Build/load tree once, then retrieve — avoids rebuilding inside compare()
    ra = build_tree(row, config=config, cache_dir=cache_dir, force_rebuild=force_rebuild)

    context, _ = ra.retrieve(
        question=RETRIEVAL_QUERY,
        top_k=20,
        max_tokens=10000,
        collapse_tree=True,
        return_layer_information=True,
    )

    return compare(
        text=text,
        reference=reference,
        title=title,
        config=config,
        raptor_context=context,
        use_judge=use_judge,
        judge_model=judge_model,
    )
