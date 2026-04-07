"""
evaluator.py
------------
Evaluates the RAPTOR summarization pipeline against QASPER documents.

Since this subagent only handles summary requests, evaluation uses two metrics:

ROUGE-L (primary)
    Measures longest common subsequence overlap between prediction and reference.
    Fast, interpretable, and standard for summarization benchmarks.

BERTScore (secondary)
    Uses contextual embeddings to compare prediction and reference semantically.
    Paraphrases score well — a high BERTScore with low ROUGE means the summary
    is semantically correct but worded differently than the abstract, which is fine.
    If both are low, the summary is genuinely missing content.
"""

from dataclasses import dataclass, field
from rouge_score import rouge_scorer
from bert_score import score as bert_score

from pipeline import Pipeline, PipelineResult


# --- Scorers ----------------------------------------------------------------

_rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


# --- Data structures --------------------------------------------------------

@dataclass
class SummaryResult:
    doc_id: str
    prediction: str
    reference: str          # document abstract used as gold summary
    rouge1: float
    rouge2: float
    rougeL: float
    bertscore_f1: float     # semantic similarity (higher = more meaning preserved)


@dataclass
class EvalReport:
    results: list[SummaryResult] = field(default_factory=list)

    @property
    def mean_rouge1(self) -> float:
        return sum(r.rouge1 for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def mean_rouge2(self) -> float:
        return sum(r.rouge2 for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def mean_rougeL(self) -> float:
        return sum(r.rougeL for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def mean_bertscore(self) -> float:
        return sum(r.bertscore_f1 for r in self.results) / len(self.results) if self.results else 0.0

    def print(self) -> None:
        print("\n=== Evaluation Report ===")
        print(f"Documents evaluated : {len(self.results)}")
        print(f"Mean ROUGE-1        : {self.mean_rouge1:.4f}")
        print(f"Mean ROUGE-2        : {self.mean_rouge2:.4f}")
        print(f"Mean ROUGE-L        : {self.mean_rougeL:.4f}")
        print(f"Mean BERTScore F1   : {self.mean_bertscore:.4f}")

        print("\nPer-document breakdown:")
        sep = "-" * 80
        for r in self.results:
            print(f"\n{sep}")
            print(f"Document : {r.doc_id}")
            print(f"Scores   : ROUGE-1={r.rouge1:.3f} | ROUGE-2={r.rouge2:.3f} | "
                  f"ROUGE-L={r.rougeL:.3f} | BERTScore={r.bertscore_f1:.3f}")
            print(f"{sep}")
            print("REFERENCE:")
            print(r.reference)
            print(f"{sep}")
            print("PREDICTION:")
            print(r.prediction)
            print(sep)

        print("\n=== Done ===\n")


# --- Evaluator --------------------------------------------------------------

class Evaluator:
    """
    Runs the summarization pipeline and scores output against the abstract.
    """

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    def evaluate_document(self, document: dict, doc_id: str = "doc") -> EvalReport:
        """
        Summarize one QASPER document and score against its abstract.

        Args:
            document: A single QASPER dataset entry.
            doc_id:   Label used in the report.

        Returns:
            EvalReport with ROUGE scores.
        """
        report = EvalReport()
        reference = document.get("abstract", "")

        if not reference:
            print(f"  [{doc_id}] No abstract found — skipping.")
            return report

        print(f"Summarizing document: {document.get('title', doc_id)}")
        result: PipelineResult = self.pipeline.run()

        rouge_scores = _rouge.score(reference, result.answer)

        print("  Computing BERTScore...")
        _, _, F1 = bert_score([result.answer], [reference], lang="en", verbose=False)

        report.results.append(SummaryResult(
            doc_id=doc_id,
            prediction=result.answer,
            reference=reference,
            rouge1=rouge_scores["rouge1"].fmeasure,
            rouge2=rouge_scores["rouge2"].fmeasure,
            rougeL=rouge_scores["rougeL"].fmeasure,
            bertscore_f1=F1[0].item(),
        ))

        return report


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
    from pipeline import Pipeline

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    dataset = load_dataset("allenai/qasper", split="train")
    document = dataset[0]
    chunks = chunk_qasper_document(document, doc_id="qasper_train_0")

    embedder = Embedder()
    summarizer = Summarizer(openai_api_key=api_key)
    tree = build_tree(chunks, embedder, summarizer)
    pipeline = Pipeline(tree, embedder, openai_api_key=api_key)

    evaluator = Evaluator(pipeline)
    report = evaluator.evaluate_document(document, doc_id="qasper_train_0")
    report.print()
