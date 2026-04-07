# QASPER RAPTOR Assistant

## WHY
A question-answering and summarization assistant over the QASPER dataset
(NLP research papers). Implements the RAPTOR architecture (ICLR 2024):
recursive embedding, clustering, and summarization to build a tree that
captures both granular details and high-level themes.
Intent detection routes summary requests to the root-level tree nodes,
and specific questions through collapsed-tree retrieval.

## WHAT
- Stack: Python, LangChain + OpenAI API, HuggingFace datasets, SBERT, UMAP, scikit-learn
- `chunker.py`   — splits QASPER documents into ~100-token sentence-preserving chunks
- `embedder.py`  — embeds chunks with SBERT (multi-qa-mpnet-base-cos-v1, 768-dim)
- `clusterer.py` — two-pass UMAP + GMM clustering with BIC-optimal cluster count
- `summarizer.py`— summarizes each cluster with an LLM (gpt-4.1-nano, Appendix D prompt)
- `tree.py`      — wires steps above into a recursive bottom-up tree builder
- `retriever.py` — collapsed-tree cosine similarity search with 2000-token budget
- `pipeline.py`  — intent detection → retrieval → answer generation
- `evaluator.py` — token-level F1 scoring against QASPER ground truth
- `main.ipynb`   — step-by-step notebook for running and verifying each component

## HOW
- Open `main.ipynb` and run cells top to bottom
- Each module has a `verify_*` function for sanity-checking that step's output
- Use `importlib.reload()` to reload a module without restarting the kernel
- Always activate venv first: `.venv\Scripts\activate`

## KNOWN LIMITATIONS
- QASPER `full_text` does not include table content — questions about numeric
  results will score 0 F1 regardless of retrieval quality
- Token counting uses whitespace splitting, not a proper tokenizer
- The GMM is fitted twice per clustering pass (once for BIC search, once for
  final assignment) — correctness is unaffected but it is slow

## Further reading
- RAPTOR paper: `RAPTOR.pdf` in the project root
- Key sections: Section 3 (Methods), Appendix D (summarization prompt),
  Appendix F (retrieval pseudocode), Appendix B (clustering ablation)
