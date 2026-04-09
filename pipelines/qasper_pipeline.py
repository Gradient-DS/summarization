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

# Project root → lib/ contains the raptor package; pipelines/ contains prompts
_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_root, "lib"))   # for `from raptor import ...`
sys.path.insert(0, os.path.dirname(__file__))    # for `from prompts import ...`

import json
import random

from openai import OpenAI
import logging

from prompts import (
    DOCUMENT_SUMMARY_SYSTEM, DOCUMENT_SUMMARY_USER,
    JUDGE_SYSTEM, JUDGE_USER,
    JUDGE_NO_REF_SYSTEM, JUDGE_NO_REF_USER,
    RETRIEVAL_QUERY,
    COD_SYSTEM, COD_USER,
)
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
        tb_num_layers=5,
        tb_max_tokens=200,            # smaller chunks → more leaf nodes → deeper tree
        tb_threshold=0.1,
        tb_top_k=20,
        tb_selection_mode="threshold",
        tb_summarization_length_ratio=1/3,
        tb_summarization_length_min=100,
        tb_summarization_length_max=400,
        tb_reduction_dimension=6,     # was 10; lower → deeper tree (stops when ≤4 nodes)
        tr_threshold=0.1,
        tr_top_k=20,
        tr_selection_mode="threshold",
    )


def chain_of_density_summarize(
    context: str,
    client: OpenAI,
    model: str = "gpt-4.1-nano",
) -> str:
    """
    Content-driven summarization inspired by Chain of Density (Adams et al., 2023).

    Rather than targeting a fixed word count, the model first extracts ALL key
    narrative elements (characters, events, decisions), then writes a summary that
    covers every one of them. Length scales naturally with the amount of important
    content in the retrieved context.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": COD_SYSTEM},
            {"role": "user", "content": COD_USER.format(context=context)},
        ],
        max_tokens=2000,
    )

    raw = response.choices[0].message.content.strip()
    try:
        data = json.loads(raw)
        summary = data.get("summary", "")
        if summary:
            n_elements = len(data.get("elements", []))
            logging.info(f"CoD extracted {n_elements} narrative elements → {len(summary.split())} word summary")
            return summary
    except (json.JSONDecodeError, KeyError, TypeError):
        logging.warning("CoD JSON parsing failed; falling back to direct summarization")

    # Fallback to single-shot if JSON parsing fails
    fallback = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": DOCUMENT_SUMMARY_SYSTEM},
            {"role": "user", "content": DOCUMENT_SUMMARY_USER.format(context=context)},
        ],
        max_tokens=1000,
    )
    return fallback.choices[0].message.content.strip()


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
        question=RETRIEVAL_QUERY,
        top_k=20,
        max_tokens=20000,
        collapse_tree=True,
        return_layer_information=True,
    )

    client = OpenAI()
    summary = chain_of_density_summarize(context, client)

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



def judge_summaries(
    text: str,
    summary_raptor: str,
    summary_baseline: str,
    reference: str = None,
    max_source_words: int = 2000,
    judge_model: str = "gpt-4.1",
) -> dict:
    """
    Use an independent LLM to pick the better of two summaries.

    The judge sees a truncated source excerpt, optionally a reference, and both
    summaries in randomised order (to avoid position bias). The result is
    un-shuffled before returning so keys always refer to the correct system.

    Args:
        text:              Original document text (will be truncated).
        summary_raptor:    Summary produced by RAPTOR.
        summary_baseline:  Summary produced by the plain-LLM baseline.
        reference:         Human-written reference summary, or None for
                           reference-free evaluation (e.g. multi-chapter texts).
        max_source_words:  How many words of the source to show the judge.
        judge_model:       Model to use as judge. Should be stronger than the
                           generator model to avoid self-preference bias.

    Returns:
        dict with keys:
          "winner"       — "raptor", "baseline", or "tie"
          "reasoning"    — judge's one-sentence justification
          "scores"       — {"raptor": {...}, "baseline": {...}} each with
                           faithfulness / coverage / conciseness (1–10)
          "reference_free" — True if no reference was provided
          "raw"          — the raw JSON string returned by the judge
    """
    source_excerpt = " ".join(text.split()[:max_source_words])
    reference_free = not reference

    # Randomise presentation order to counter position bias
    if random.random() < 0.5:
        sum_a, sum_b = summary_raptor, summary_baseline
        key_a, key_b = "raptor", "baseline"
    else:
        sum_a, sum_b = summary_baseline, summary_raptor
        key_a, key_b = "baseline", "raptor"

    if reference_free:
        system_prompt = JUDGE_NO_REF_SYSTEM
        user_prompt   = JUDGE_NO_REF_USER.format(
            source=source_excerpt,
            label_a="A", summary_a=sum_a,
            label_b="B", summary_b=sum_b,
        )
    else:
        system_prompt = JUDGE_SYSTEM
        user_prompt   = JUDGE_USER.format(
            source=source_excerpt,
            reference=reference,
            label_a="A", summary_a=sum_a,
            label_b="B", summary_b=sum_b,
        )

    client = OpenAI()
    response = client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()
    try:
        verdict = json.loads(raw)
    except json.JSONDecodeError:
        return {"winner": "error", "reasoning": "judge returned invalid JSON", "scores": {}, "reference_free": reference_free, "raw": raw}

    # Map A/B back to raptor/baseline
    scores = {
        key_a: verdict["scores"].get("A", {}),
        key_b: verdict["scores"].get("B", {}),
    }
    raw_winner = verdict.get("winner", "tie")
    if raw_winner == "A":
        winner = key_a
    elif raw_winner == "B":
        winner = key_b
    else:
        winner = "tie"

    return {
        "winner": winner,
        "reasoning": verdict.get("reasoning", ""),
        "scores": scores,
        "reference_free": reference_free,
        "raw": raw,
    }


def compare(
    text: str,
    reference: str,
    title: str = "",
    config: RetrievalAugmentationConfig = None,
    max_words: int = 7500,
    raptor_context: str = None,
    use_judge: bool = True,
    judge_model: str = "gpt-4.1",
) -> dict:
    """
    Run both RAPTOR and the baseline on the same text, evaluate against
    reference, and print a side-by-side comparison.

    Dataset-agnostic: accepts raw text + reference string directly so it
    works for both QASPER and BookSum (or any other dataset).

    Args:
        text:           Full document/chapter text to summarise.
        reference:      Reference summary to score against.
        title:          Optional document title for display.
        config:         Optional pre-built RetrievalAugmentationConfig.
        max_words:      Word cap for the baseline truncation.
        raptor_context: Pre-retrieved RAPTOR context. If provided, skips
                        tree building and retrieval (uses cached result).

    Returns:
        dict with keys "raptor" and "baseline", each an EvalScores dict.
    """
    if title:
        print(f"Document: {title}")

    client = OpenAI()

    # --- Baseline ---
    print("Running baseline...")
    truncated = " ".join(text.split()[:max_words])
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": DOCUMENT_SUMMARY_SYSTEM},
            {"role": "user", "content": DOCUMENT_SUMMARY_USER.format(context=truncated)},
        ],
        max_tokens=500,
    )
    baseline_summary = response.choices[0].message.content.strip()
    baseline_scores  = evaluate_summary(baseline_summary, reference) if reference else None

    # --- RAPTOR ---
    if raptor_context is None:
        print("Running RAPTOR...")
        if config is None:
            config = build_raptor_config()
        ra = RetrievalAugmentation(config=config)
        ra.add_documents(text)
        raptor_context, _ = ra.retrieve(
            question=RETRIEVAL_QUERY,
            top_k=20,
            max_tokens=10000,
            collapse_tree=True,
            return_layer_information=True,
        )
    else:
        print("Using pre-built RAPTOR context...")

    print("Generating RAPTOR summary (Chain of Density)...")
    raptor_summary = chain_of_density_summarize(raptor_context, client)
    raptor_scores  = evaluate_summary(raptor_summary, reference) if reference else None

    # --- Judge ---
    judgment = None
    if use_judge:
        print("Running LLM judge...")
        judgment = judge_summaries(
            text=text,
            summary_raptor=raptor_summary,
            summary_baseline=baseline_summary,
            reference=reference or None,   # pass None explicitly when empty
            judge_model=judge_model,
        )

    # --- Print ---
    sep = "=" * 80
    thin = "-" * 80
    print(f"\n{sep}")
    print("BASELINE SUMMARY (plain LLM, no RAPTOR):")
    print(thin)
    print(baseline_summary)
    print(f"\n{sep}")
    print("RAPTOR SUMMARY (tree retrieval):")
    print(thin)
    print(raptor_summary)
    if reference:
        print(f"\n{sep}")
        print("REFERENCE:")
        print(thin)
        print(reference)
        print(f"\n{sep}")
        print(f"{'METRIC':<25} {'BASELINE':>10} {'RAPTOR':>10} {'DELTA':>10}")
        print(thin)
        for k in raptor_scores:
            delta = raptor_scores[k] - baseline_scores[k]
            sign = "+" if delta >= 0 else ""
            print(f"  {k:<23} {baseline_scores[k]:>10.4f} {raptor_scores[k]:>10.4f} {sign}{delta:>9.4f}")
    else:
        print(f"\n{sep}")
        print("(No reference summary available — ROUGE/BERTScore skipped)")

    if judgment:
        ref_note = " — reference-free" if judgment.get("reference_free") else ""
        print(f"\n{'LLM JUDGE' + ref_note:^80}")
        print(thin)
        print(f"  Winner   : {judgment['winner'].upper()}")
        print(f"  Reasoning: {judgment['reasoning']}")
        if judgment["scores"]:
            print(f"\n  {'DIMENSION':<20} {'BASELINE':>10} {'RAPTOR':>10}")
            for dim in ["faithfulness", "coverage", "conciseness"]:
                b = judgment["scores"].get("baseline", {}).get(dim, "-")
                r = judgment["scores"].get("raptor",   {}).get(dim, "-")
                print(f"  {dim:<20} {str(b):>10} {str(r):>10}")

    print(sep)

    result = {"raptor": raptor_scores, "baseline": baseline_scores}
    if judgment:
        result["judgment"] = judgment
    return result


def compare_document(document: dict, config: RetrievalAugmentationConfig = None) -> dict:
    """Convenience wrapper for QASPER documents."""
    return compare(
        text=extract_qasper_text(document),
        reference=document.get("abstract", ""),
        title=document.get("title", ""),
        config=config,
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
