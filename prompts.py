"""
prompts.py
----------
All LLM prompts used in the QASPER summarization pipeline.
Centralised here so they can be versioned and swapped independently of logic.
"""

# --- Tree construction (RAPTOR paper Appendix D) ---

SUMMARIZATION_SYSTEM = "You are a Summarizing Text Portal"

SUMMARIZATION_USER = (
    "Write a summary of the following, including as many key details as possible: {context}:"
)

# --- Final document summary (pipeline output) ---

DOCUMENT_SUMMARY_SYSTEM = "You are a document summarization assistant."

DOCUMENT_SUMMARY_USER = (
    "Write a concise but complete summary of the document based on "
    "the following content:\n\n{context}"
)
