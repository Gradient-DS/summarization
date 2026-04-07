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
import sys

_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_root, "lib"))   # for `from raptor import ...`
sys.path.insert(0, os.path.dirname(__file__))    # for `from prompts/qasper_pipeline import ...`

from qasper_pipeline import (
    RetrievalAugmentationConfig,
    SummaryResult,
    EvalScores,
    build_raptor_config,
    evaluate_summary,
    print_evaluation,
    compare,
    summarize_document as _summarize_text,
)
from openai import OpenAI
from prompts import DOCUMENT_SUMMARY_SYSTEM, DOCUMENT_SUMMARY_USER
from raptor import RetrievalAugmentation


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

def summarize_chapter(row: dict, config: RetrievalAugmentationConfig = None) -> SummaryResult:
    """
    Build a RAPTOR tree from a BookSum chapter and generate a summary.

    Args:
        row:    A single BookSum dataset row.
        config: Optional pre-built RetrievalAugmentationConfig.

    Returns:
        SummaryResult with title, abstract (=summary_text), summary, scores={}.
    """
    text, reference, title = extract_booksum_text(row)

    if config is None:
        config = build_raptor_config()

    ra = RetrievalAugmentation(config=config)
    ra.add_documents(text)

    context, _ = ra.retrieve(
        question="Summarize this chapter",
        top_k=10,
        max_tokens=3500,
        collapse_tree=True,
        return_layer_information=True,
    )

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": DOCUMENT_SUMMARY_SYSTEM},
            {"role": "user", "content": DOCUMENT_SUMMARY_USER.format(context=context)},
        ],
        max_tokens=500,
    )
    summary = response.choices[0].message.content.strip()

    return SummaryResult(title=title, abstract=reference, summary=summary, scores={})


# ---------------------------------------------------------------------------
# Query a chapter (the "What does this book say about X?" use case)
# ---------------------------------------------------------------------------

def query_chapter(row: dict, question: str, config: RetrievalAugmentationConfig = None) -> str:
    """
    Answer a question about a BookSum chapter using RAPTOR tree retrieval.

    Example:
        query_chapter(row, "What does this chapter say about friendship?")

    Args:
        row:      A single BookSum dataset row.
        question: The question to answer.
        config:   Optional pre-built RetrievalAugmentationConfig.

    Returns:
        The answer string.
    """
    text, _, _ = extract_booksum_text(row)

    if config is None:
        config = build_raptor_config()

    ra = RetrievalAugmentation(config=config)
    ra.add_documents(text)

    context, _ = ra.retrieve(
        question=question,
        top_k=10,
        max_tokens=3500,
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

def compare_chapter(row: dict, config: RetrievalAugmentationConfig = None) -> dict:
    """Convenience wrapper: run RAPTOR vs baseline comparison on a BookSum row."""
    text, reference, title = extract_booksum_text(row)
    return compare(text=text, reference=reference, title=title, config=config)
