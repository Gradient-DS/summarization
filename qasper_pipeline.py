"""
qasper_pipeline.py
------------------
QASPER-specific helpers that sit on top of the official RAPTOR library.

Responsibilities:
  1. extract_qasper_text()   — pull structured text out of a QASPER record
  2. summarize_document()    — build a RAPTOR tree and generate a summary
  3. evaluate_summary()      — score a summary with ROUGE-1/2/L and BERTScore
"""

import os
import sys
from typing import TypedDict

# Make the raptor package importable from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "raptor"))

from openai import OpenAI
from prompts import DOCUMENT_SUMMARY_SYSTEM, DOCUMENT_SUMMARY_USER
from raptor import (
    GPT4NanoSummarizationModel,
    RetrievalAugmentation,
    RetrievalAugmentationConfig,
    SBertEmbeddingModel,
)


# ---------------------------------------------------------------------------
# Typed output
# ---------------------------------------------------------------------------

class SummaryResult(TypedDict):
    title: str
    abstract: str
    summary: str
    scores: dict  # populated after evaluate_summary(); empty until then


class EvalScores(TypedDict):
    rouge1: float
    rouge2: float
    rougeL: float
    bertscore_precision: float
    bertscore_recall: float
    bertscore_f1: float


# ---------------------------------------------------------------------------
# 1. Text extraction
# ---------------------------------------------------------------------------

def extract_qasper_text(document: dict) -> str:
    """
    Extract all readable text from a QASPER dataset record and return it as
    a single string suitable for passing to RAPTOR's add_documents().

    QASPER structure:
        {
          "title":     str,
          "abstract":  str,
          "full_text": {
              "section_name": [str, ...],
              "paragraphs":   [[str, ...], ...]   # one list per section
          }
        }
    """
    parts = []

    if document.get("title"):
        parts.append(document["title"])

    if document.get("abstract"):
        parts.append(document["abstract"])

    full_text = document.get("full_text", {})
    section_names = full_text.get("section_name", [])
    paragraphs_per_section = full_text.get("paragraphs", [])

    for section_name, paragraphs in zip(section_names, paragraphs_per_section):
        section_text = " ".join(p for p in paragraphs if p and p.strip())
        if section_text.strip():
            parts.append(section_text)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# 2. Build tree + summarize
# ---------------------------------------------------------------------------

def build_raptor_config() -> RetrievalAugmentationConfig:
    """
    Return a RetrievalAugmentationConfig wired with:
      - gpt-4.1-nano for summarization
      - SBERT (multi-qa-mpnet-base-cos-v1) for embeddings
    """
    return RetrievalAugmentationConfig(
        summarization_model=GPT4NanoSummarizationModel(),
        embedding_model=SBertEmbeddingModel(),
    )


def summarize_document(
    document: dict,
    config: RetrievalAugmentationConfig = None,
) -> SummaryResult:
    """
    Build a RAPTOR tree from a QASPER document and generate a summary.

    Args:
        document: A single QASPER dataset record.
        config:   Optional pre-built RetrievalAugmentationConfig.
                  If None, uses build_raptor_config().

    Returns:
        SummaryResult with title, abstract, summary, and empty scores dict.
    """
    if config is None:
        config = build_raptor_config()

    text = extract_qasper_text(document)
    abstract = document.get("abstract", "")
    title = document.get("title", "")

    ra = RetrievalAugmentation(config=config)
    ra.add_documents(text)

    context, _ = ra.retrieve(
        question="Summarize this document",
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

    return SummaryResult(title=title, abstract=abstract, summary=summary, scores={})


# ---------------------------------------------------------------------------
# 3. Evaluation
# ---------------------------------------------------------------------------

def evaluate_summary(prediction: str, reference: str) -> EvalScores:
    """
    Compute ROUGE-1/2/L and BERTScore (roberta-large) for a single
    prediction/reference pair.
    """
    from rouge_score import rouge_scorer
    from bert_score import score as bert_score

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge = scorer.score(reference, prediction)

    P, R, F = bert_score(
        [prediction], [reference],
        lang="en", model_type="roberta-large", verbose=False,
    )

    return EvalScores(
        rouge1=rouge["rouge1"].fmeasure,
        rouge2=rouge["rouge2"].fmeasure,
        rougeL=rouge["rougeL"].fmeasure,
        bertscore_precision=P[0].item(),
        bertscore_recall=R[0].item(),
        bertscore_f1=F[0].item(),
    )


def print_evaluation(prediction: str, reference: str) -> EvalScores:
    """Evaluate and print results with clear formatting."""
    sep = "-" * 80
    print("REFERENCE:")
    print(reference)
    print(sep)
    print("PREDICTION:")
    print(prediction)
    print(sep)

    scores = evaluate_summary(prediction, reference)
    print("SCORES:")
    print(f"  ROUGE-1  : {scores['rouge1']:.4f}")
    print(f"  ROUGE-2  : {scores['rouge2']:.4f}")
    print(f"  ROUGE-L  : {scores['rougeL']:.4f}")
    print(f"  BERTScore: P={scores['bertscore_precision']:.4f}  "
          f"R={scores['bertscore_recall']:.4f}  "
          f"F1={scores['bertscore_f1']:.4f}")
    return scores
