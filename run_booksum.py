"""
run_booksum.py
--------------
Batch evaluation of RAPTOR vs. plain-LLM baseline on BookSum chapters,
with an optional LLM-as-judge verdict for each pair.

Usage:
    python run_booksum.py                        # 10 chapters, validation split
    python run_booksum.py --n 5                  # 5 chapters
    python run_booksum.py --split train          # use train split
    python run_booksum.py --no-judge             # skip LLM judge (cheaper)
    python run_booksum.py --force-rebuild        # ignore cached trees
    python run_booksum.py --judge-model gpt-4o   # use a different judge model

Trees are cached in .tree_cache/ so repeated runs skip the expensive build step.
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_root, "lib"))
sys.path.insert(0, os.path.join(_root, "pipelines"))

from dotenv import load_dotenv

load_dotenv()

import numpy as np
from booksum_pipeline import (
    build_raptor_config,
    compare_chapter,
    extract_booksum_text,
    load_booksum,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _short_title(title: str, width: int = 45) -> str:
    return title if len(title) <= width else title[: width - 1] + "…"


def _print_per_doc_table(rows: list[dict]) -> None:
    """Print a per-document result table."""
    has_scores = any(r["raptor"] is not None for r in rows)
    sep  = "=" * 110
    thin = "-" * 110
    print(f"\n{sep}")
    print("PER-DOCUMENT RESULTS  (B- = baseline)")
    print(thin)
    if has_scores:
        print(f"  {'#':>3}  {'TITLE':<46} {'ROUGE-L':>8} {'B-ROUGE-L':>9} {'BERT-F1':>8} {'B-BERT-F1':>9} {'JUDGE':>8}")
    else:
        print(f"  {'#':>3}  {'TITLE':<46} {'JUDGE':>8}  REASONING")
    print(thin)
    for r in rows:
        title  = _short_title(r["title"])
        winner = r.get("judgment", {}).get("winner", "n/a").upper()
        if has_scores and r["raptor"]:
            rougeL  = r["raptor"]["rougeL"]
            brougeL = r["baseline"]["rougeL"]
            bertf1  = r["raptor"]["bertscore_f1"]
            bbertf1 = r["baseline"]["bertscore_f1"]
            print(f"  {r['i']:>3}  {title:<46} {rougeL:>8.4f} {brougeL:>9.4f} {bertf1:>8.4f} {bbertf1:>9.4f} {winner:>8}")
        else:
            reasoning = r.get("judgment", {}).get("reasoning", "")[:55]
            print(f"  {r['i']:>3}  {title:<46} {winner:>8}  {reasoning}")
    print(sep)


def _print_aggregate(rows: list[dict]) -> None:
    """Print aggregate metrics and judge win counts."""
    metrics = ["rouge1", "rouge2", "rougeL", "bertscore_f1"]
    sep  = "=" * 70
    thin = "-" * 70

    scored_rows = [r for r in rows if r["raptor"] is not None]
    print(f"\n{sep}")
    print("AGGREGATE AVERAGES")
    print(thin)
    if scored_rows:
        print(f"  {'METRIC':<25} {'BASELINE':>10} {'RAPTOR':>10} {'DELTA':>10}")
        print(thin)
        for m in metrics:
            b_vals = [r["baseline"][m] for r in scored_rows]
            r_vals = [r["raptor"][m]   for r in scored_rows]
            b_avg  = np.mean(b_vals)
            r_avg  = np.mean(r_vals)
            delta  = r_avg - b_avg
            sign   = "+" if delta >= 0 else ""
            print(f"  {m:<25} {b_avg:>10.4f} {r_avg:>10.4f} {sign}{delta:>9.4f}")
    else:
        print("  (No reference summaries — ROUGE/BERTScore not available)")

    # Judge win tally
    judgments = [r["judgment"] for r in rows if "judgment" in r]
    if judgments:
        raptor_wins   = sum(1 for j in judgments if j["winner"] == "raptor")
        baseline_wins = sum(1 for j in judgments if j["winner"] == "baseline")
        ties          = sum(1 for j in judgments if j["winner"] == "tie")
        errors        = sum(1 for j in judgments if j["winner"] == "error")
        total         = len(judgments)
        print(thin)
        print(f"  LLM JUDGE ({total} chapters judged):")
        print(f"    RAPTOR wins   : {raptor_wins:>3}  ({100*raptor_wins/total:.0f}%)")
        print(f"    Baseline wins : {baseline_wins:>3}  ({100*baseline_wins/total:.0f}%)")
        print(f"    Ties          : {ties:>3}  ({100*ties/total:.0f}%)")
        if errors:
            print(f"    Errors        : {errors:>3}")

        # Average judge scores per dimension
        dims = ["faithfulness", "coverage", "conciseness"]
        print(thin)
        print(f"  {'JUDGE DIMENSION':<25} {'BASELINE':>10} {'RAPTOR':>10}")
        print(thin)
        for dim in dims:
            b_scores = [j["scores"].get("baseline", {}).get(dim) for j in judgments]
            r_scores = [j["scores"].get("raptor",   {}).get(dim) for j in judgments]
            b_scores = [s for s in b_scores if s is not None]
            r_scores = [s for s in r_scores if s is not None]
            if b_scores and r_scores:
                print(f"  {dim:<25} {np.mean(b_scores):>10.2f} {np.mean(r_scores):>10.2f}")

    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="BookSum batch evaluation: RAPTOR vs baseline")
    parser.add_argument("--n",             type=int,  default=10,           help="Number of chapters to evaluate (default: 10)")
    parser.add_argument("--split",         type=str,  default="validation",  help="Dataset split: train / validation / test (default: validation)")
    parser.add_argument("--no-judge",      action="store_true",             help="Skip LLM-as-judge step")
    parser.add_argument("--judge-model",   type=str,  default="gpt-4.1",    help="Model to use as judge (default: gpt-4.1)")
    parser.add_argument("--force-rebuild", action="store_true",             help="Ignore cached trees and rebuild from scratch")
    parser.add_argument("--aggregate",     action="store_true",             help="Use multi-chapter aggregate rows instead of single chapters (better tests RAPTOR's long-doc advantage)")
    args = parser.parse_args()

    use_judge = not args.no_judge
    n         = args.n

    # single_chapters_only=True  → short chapters  (baseline tends to win)
    # single_chapters_only=False + is_aggregate     → long multi-chapter texts (RAPTOR's strength)
    if args.aggregate:
        print(f"Loading BookSum ({args.split} split, aggregate/multi-chapter rows)...")
        from datasets import load_dataset
        import logging, warnings
        warnings.filterwarnings("ignore")
        logging.disable(logging.CRITICAL)
        ds_raw = load_dataset("kmfoda/booksum", split=args.split)
        logging.disable(logging.NOTSET)
        ds = ds_raw.filter(lambda r: bool(r["is_aggregate"]) and bool(r["chapter"] and r["chapter"].strip()))
        mode_label = "aggregate (multi-chapter)"
    else:
        print(f"Loading BookSum ({args.split} split, single chapters)...")
        ds = load_booksum(split=args.split)
        mode_label = "single chapters"

    print(f"Mode: {mode_label} — {len(ds)} rows available")
    if n > len(ds):
        print(f"Warning: only {len(ds)} chapters available, evaluating all.")
        n = len(ds)

    config = build_raptor_config()
    rows   = []

    for i in range(n):
        row = ds[i]
        _, _, title = extract_booksum_text(row)
        print(f"\n[{i+1}/{n}] {title}")

        try:
            result = compare_chapter(
                row,
                config=config,
                force_rebuild=args.force_rebuild,
                use_judge=use_judge,
                judge_model=args.judge_model,
            )
            result["i"]     = i + 1
            result["title"] = title
            rows.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    if not rows:
        print("No results to display.")
        return

    _print_per_doc_table(rows)
    _print_aggregate(rows)


if __name__ == "__main__":
    main()
