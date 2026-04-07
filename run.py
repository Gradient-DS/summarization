"""
run.py
------
CLI entry point for the QASPER summarization pipeline.
Called by the general agent when a summarization request is detected,
or run directly for testing.

Usage:
    python run.py                        # summarize document 0 from validation split
    python run.py --index 3              # summarize document at index 3
    python run.py --split train          # use train split
    python run.py --batch 5             # evaluate first 5 documents
"""

import argparse
import os

from dotenv import load_dotenv

load_dotenv()


def run_single(index: int, split: str) -> None:
    import warnings
    warnings.filterwarnings("ignore")

    from datasets import load_dataset
    from qasper_pipeline import build_raptor_config, print_evaluation, summarize_document

    print(f"Loading QASPER ({split} split, index {index})...")
    dataset = load_dataset("allenai/qasper", split=split)
    document = dataset[index]

    print(f"Title: {document['title']}\n")
    config = build_raptor_config()

    print("Building RAPTOR tree and generating summary...")
    result = summarize_document(document, config=config)

    print_evaluation(result["summary"], result["abstract"])


def run_batch(n: int, split: str) -> None:
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    from datasets import load_dataset
    from qasper_pipeline import build_raptor_config, evaluate_summary, summarize_document

    print(f"Batch evaluation: {n} documents from {split} split\n")
    dataset = load_dataset("allenai/qasper", split=split)
    config = build_raptor_config()
    all_scores = []

    for i in range(n):
        doc = dataset[i]
        print(f"[{i+1}/{n}] {doc['title'][:60]}")
        result = summarize_document(doc, config=config)
        scores = evaluate_summary(result["summary"], result["abstract"])
        all_scores.append(scores)
        print(f"  ROUGE-L={scores['rougeL']:.3f}  BERTScore-F1={scores['bertscore_f1']:.3f}")

    print("\n--- AVERAGE ---")
    for k in all_scores[0]:
        print(f"  {k}: {np.mean([s[k] for s in all_scores]):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QASPER RAPTOR summarization pipeline")
    parser.add_argument("--index", type=int, default=0, help="Document index (default: 0)")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split (default: validation)")
    parser.add_argument("--batch", type=int, default=None, help="Evaluate N documents in batch mode")
    args = parser.parse_args()

    if args.batch:
        run_batch(args.batch, args.split)
    else:
        run_single(args.index, args.split)
