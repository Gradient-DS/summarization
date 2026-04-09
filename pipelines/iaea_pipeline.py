"""
iaea_pipeline.py
----------------
Pipeline for IAEA (and similar) document collections stored as a JSON metadata file.

Expected JSON structure:
    {
      "metadata": { "source": "iaea", "total_count": 356, ... },
      "items": [
        {
          "source_id": "58797451",
          "download_url": "https://...",
          "title": "GSG-1 Classification of Radioactive Waste",
          "filetype": "PDF",
          "year": 2009,
          ...
        },
        ...
      ]
    }

Workflow:
  1. load_iaea_items()       — parse the JSON metadata file
  2. download_pdf()          — download and cache PDF locally
  3. extract_pdf_text()      — extract plain text from the PDF
  4. generate_golden_standard() — use gpt-4.1 to create a reference summary
  5. compare_iaea_document() — run RAPTOR vs baseline, score against the golden standard
"""

import json
import logging
import os
import re
import sys
from typing import Optional

import requests

_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_root, "lib"))
sys.path.insert(0, os.path.dirname(__file__))

from openai import OpenAI
from prompts import (
    TECH_SUMMARIZATION_SYSTEM, TECH_SUMMARIZATION_USER,
    TECH_RETRIEVAL_QUERY, TECH_COD_SYSTEM, TECH_COD_USER,
    GOLDEN_STANDARD_SYSTEM, GOLDEN_STANDARD_USER,
    JUDGE_SYSTEM, JUDGE_USER, JUDGE_NO_REF_SYSTEM, JUDGE_NO_REF_USER,
)
from qasper_pipeline import (
    RetrievalAugmentationConfig,
    EvalScores,
    evaluate_summary,
    judge_summaries,
    compare,
    chain_of_density_summarize,
)
from raptor import (
    CustomPromptSummarizationModel,
    RetrievalAugmentation,
    SBertEmbeddingModel,
)

DEFAULT_PDF_CACHE = os.path.join(_root, ".pdf_cache")
DEFAULT_TREE_CACHE = os.path.join(_root, ".tree_cache")


# ---------------------------------------------------------------------------
# 1. Load metadata
# ---------------------------------------------------------------------------

def load_iaea_items(json_path: str) -> list[dict]:
    """
    Load the IAEA metadata JSON and return the list of document items.

    Args:
        json_path: Path to the JSON file produced by the scraper.

    Returns:
        List of item dicts, each with keys: title, download_url, filetype, year, etc.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("items", [])
    logging.info(f"Loaded {len(items)} items from {json_path}")
    return items


# ---------------------------------------------------------------------------
# 2. Download PDFs
# ---------------------------------------------------------------------------

def _safe_filename(title: str, source_id: str) -> str:
    safe = re.sub(r"[^\w\-]", "_", title)[:60]
    return f"{safe}_{source_id}.pdf"


def download_pdf(
    item: dict,
    cache_dir: str = DEFAULT_PDF_CACHE,
    force: bool = False,
    timeout: int = 60,
) -> Optional[str]:
    """
    Download the PDF for an IAEA item and cache it locally.

    Args:
        item:      A single item dict from load_iaea_items().
        cache_dir: Directory to store downloaded PDFs.
        force:     Re-download even if the file exists.
        timeout:   Request timeout in seconds.

    Returns:
        Local file path, or None if the download failed.
    """
    if item.get("filetype", "").upper() != "PDF":
        logging.warning(f"Skipping non-PDF item: {item.get('title')}")
        return None

    url = item.get("download_url")
    if not url:
        logging.warning(f"No download_url for: {item.get('title')}")
        return None

    os.makedirs(cache_dir, exist_ok=True)
    filename = _safe_filename(item.get("title", "untitled"), item.get("source_id", ""))
    path = os.path.join(cache_dir, filename)

    if not force and os.path.exists(path):
        logging.info(f"Using cached PDF: {path}")
        return path

    logging.info(f"Downloading: {item.get('title')} from {url}")
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        with open(path, "wb") as f:
            f.write(resp.content)
        logging.info(f"Saved to: {path}")
        return path
    except Exception as e:
        logging.error(f"Download failed for {item.get('title')}: {e}")
        return None


# ---------------------------------------------------------------------------
# 3. Extract text from PDF
# ---------------------------------------------------------------------------

def extract_pdf_text(pdf_path: str, max_pages: int = None) -> str:
    """
    Extract plain text from a PDF using pypdf.

    Args:
        pdf_path:  Path to the local PDF file.
        max_pages: Optional cap on the number of pages to extract.

    Returns:
        Extracted text as a single string. Returns empty string on failure.

    Requires: pip install pypdf
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pypdf is required: pip install pypdf")

    reader = PdfReader(pdf_path)
    pages = reader.pages[:max_pages] if max_pages else reader.pages

    parts = []
    for i, page in enumerate(pages):
        text = page.extract_text() or ""
        if text.strip():
            parts.append(text)

    full_text = "\n\n".join(parts)
    logging.info(f"Extracted {len(reader.pages)} pages → {len(full_text.split())} words from {pdf_path}")
    return full_text


# ---------------------------------------------------------------------------
# 4. Golden standard
# ---------------------------------------------------------------------------

def generate_golden_standard(
    text: str,
    title: str,
    model: str = "gpt-5.4",
    max_words: int = 20000,
) -> str:
    """
    Use a strong LLM to generate a reference summary of the document.
    This serves as the gold standard when no human-written reference exists.

    The full text is truncated to max_words to stay within context limits.
    For very long documents consider chunking and multi-pass summarization.

    Args:
        text:      Extracted document text.
        title:     Document title (shown to the model for context).
        model:     Model to use — should be the strongest available (default gpt-4.1).
        max_words: Word cap for the input text.

    Returns:
        Reference summary string.
    """
    truncated = " ".join(text.split()[:max_words])
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": GOLDEN_STANDARD_SYSTEM},
            {"role": "user", "content": GOLDEN_STANDARD_USER.format(
                title=title, text=truncated
            )},
        ],
        max_completion_tokens=8000,
    )
    choice = response.choices[0]
    # Some newer models (o-series, gpt-5.x) may surface content via reasoning output
    # when the standard content field is None/empty.
    content = choice.message.content
    if not content:
        # Try the refusal field first, then fall back to the raw response text
        content = getattr(choice.message, "refusal", None)
    if not content:
        logging.warning(f"Empty content from {model}. Full choice: {choice}")
        content = ""
    return content.strip()


# ---------------------------------------------------------------------------
# 5. RAPTOR config for technical documents
# ---------------------------------------------------------------------------

def build_iaea_config() -> RetrievalAugmentationConfig:
    """
    RetrievalAugmentationConfig tuned for technical/regulatory documents:
      - CustomPromptSummarizationModel with technical prompts (not literary)
      - SBERT embeddings
      - Smaller chunks (150 tokens) for dense regulatory text
    """
    return RetrievalAugmentationConfig(
        summarization_model=CustomPromptSummarizationModel(
            system_prompt=TECH_SUMMARIZATION_SYSTEM,
            user_prompt=TECH_SUMMARIZATION_USER,
        ),
        embedding_model=SBertEmbeddingModel(),
        tb_num_layers=5,
        tb_max_tokens=150,
        tb_threshold=0.1,
        tb_top_k=20,
        tb_selection_mode="threshold",
        tb_summarization_length_ratio=1/3,
        tb_summarization_length_min=100,
        tb_summarization_length_max=400,
        tb_reduction_dimension=6,
        tr_threshold=0.1,
        tr_top_k=20,
        tr_selection_mode="threshold",
    )


# ---------------------------------------------------------------------------
# 6. Build RAPTOR tree (with caching)
# ---------------------------------------------------------------------------

def build_tree(
    text: str,
    title: str,
    source_id: str,
    config: RetrievalAugmentationConfig = None,
    cache_dir: str = DEFAULT_TREE_CACHE,
    force_rebuild: bool = False,
) -> RetrievalAugmentation:
    """
    Build (or load from cache) a RAPTOR tree for an IAEA document.
    """
    if config is None:
        config = build_iaea_config()

    os.makedirs(cache_dir, exist_ok=True)
    safe = re.sub(r"[^\w\-]", "_", title)[:60]
    path = os.path.join(cache_dir, f"iaea_{source_id}_{safe}.pkl")

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
# 7. Full comparison pipeline
# ---------------------------------------------------------------------------

def compare_iaea_document(
    item: dict,
    json_path: str = None,
    pdf_path: str = None,
    text: str = None,
    golden_standard: str = None,
    config: RetrievalAugmentationConfig = None,
    pdf_cache_dir: str = DEFAULT_PDF_CACHE,
    tree_cache_dir: str = DEFAULT_TREE_CACHE,
    force_rebuild: bool = False,
    golden_model: str = "gpt-4.1",
    judge_model: str = "gpt-4.1",
    use_judge: bool = True,
    max_words_baseline: int = 7500,
) -> dict:
    """
    Full RAPTOR vs baseline comparison for a single IAEA document.

    Supply either:
      - pdf_path: path to an already-downloaded PDF
      - text: pre-extracted plain text
    If neither is supplied, the PDF is downloaded using item["download_url"].

    If golden_standard is None, it is generated automatically using golden_model.

    Returns:
        dict with keys "raptor", "baseline", and optionally "judgment".
    """
    title = item.get("title", "untitled")

    # --- Get text ---
    if text is None:
        if pdf_path is None:
            pdf_path = download_pdf(item, cache_dir=pdf_cache_dir)
        if pdf_path is None:
            raise RuntimeError(f"Could not obtain PDF for: {title}")
        text = extract_pdf_text(pdf_path)

    if not text.strip():
        raise ValueError(f"No text extracted from: {title}")

    # --- Golden standard ---
    if golden_standard is None:
        print(f"Generating golden standard for: {title}")
        golden_standard = generate_golden_standard(text, title, model=golden_model)

    # --- Build/load tree and retrieve context ---
    if config is None:
        config = build_iaea_config()

    ra = build_tree(
        text=text,
        title=title,
        source_id=item.get("source_id", "unknown"),
        config=config,
        cache_dir=tree_cache_dir,
        force_rebuild=force_rebuild,
    )

    raptor_context, _ = ra.retrieve(
        question=TECH_RETRIEVAL_QUERY,
        top_k=20,
        max_tokens=10000,
        collapse_tree=True,
        return_layer_information=True,
    )

    # --- Run comparison using the shared compare() with technical CoD ---
    client = OpenAI()

    # Override the final summary generation to use technical CoD prompts
    print("Running baseline...")
    truncated = " ".join(text.split()[:max_words_baseline])
    from prompts import DOCUMENT_SUMMARY_SYSTEM, DOCUMENT_SUMMARY_USER
    baseline_response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": TECH_SUMMARIZATION_SYSTEM},
            {"role": "user", "content": TECH_SUMMARIZATION_USER.format(context=truncated)},
        ],
        max_tokens=800,
    )
    baseline_summary = baseline_response.choices[0].message.content.strip()
    baseline_scores = evaluate_summary(baseline_summary, golden_standard)

    print("Generating RAPTOR summary (technical CoD)...")
    # Temporarily swap CoD prompts for technical domain
    import prompts as _prompts
    _orig_cod_system = _prompts.COD_SYSTEM
    _orig_cod_user = _prompts.COD_USER
    _prompts.COD_SYSTEM = TECH_COD_SYSTEM
    _prompts.COD_USER = TECH_COD_USER
    try:
        raptor_summary = chain_of_density_summarize(raptor_context, client)
    finally:
        _prompts.COD_SYSTEM = _orig_cod_system
        _prompts.COD_USER = _orig_cod_user

    raptor_scores = evaluate_summary(raptor_summary, golden_standard)

    # --- Judge ---
    judgment = None
    if use_judge:
        print("Running LLM judge...")
        judgment = judge_summaries(
            text=text,
            summary_raptor=raptor_summary,
            summary_baseline=baseline_summary,
            reference=golden_standard,
            judge_model=judge_model,
        )

    # --- Print ---
    sep = "=" * 80
    thin = "-" * 80
    print(f"\n{sep}")
    print(f"Document: {title} ({item.get('year', '?')})")
    print(f"{sep}")
    print("BASELINE SUMMARY:")
    print(thin)
    print(baseline_summary)
    print(f"\n{sep}")
    print("RAPTOR SUMMARY:")
    print(thin)
    print(raptor_summary)
    print(f"\n{sep}")
    print("GOLDEN STANDARD (LLM reference):")
    print(thin)
    print(golden_standard)
    print(f"\n{sep}")
    print(f"{'METRIC':<25} {'BASELINE':>10} {'RAPTOR':>10} {'DELTA':>10}")
    print(thin)
    for k in raptor_scores:
        delta = raptor_scores[k] - baseline_scores[k]
        sign = "+" if delta >= 0 else ""
        print(f"  {k:<23} {baseline_scores[k]:>10.4f} {raptor_scores[k]:>10.4f} {sign}{delta:>9.4f}")

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
                r = judgment["scores"].get("raptor", {}).get(dim, "-")
                print(f"  {dim:<20} {str(b):>10} {str(r):>10}")
    print(sep)

    result = {"raptor": raptor_scores, "baseline": baseline_scores, "golden_standard": golden_standard}
    if judgment:
        result["judgment"] = judgment
    return result
