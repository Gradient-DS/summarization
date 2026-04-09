"""
prompts.py
----------
All LLM prompts used in the QASPER summarization pipeline.
Centralised here so they can be versioned and swapped independently of logic.
"""

# --- Tree construction (RAPTOR paper Appendix D) ---

SUMMARIZATION_SYSTEM = (
    "You are a literary summarizer. Your job is to compress text while preserving "
    "concrete narrative content: who does what, key plot events, character introductions, "
    "specific outcomes, and named details. Do not replace specific events with abstract "
    "thematic statements. Do not pad with literary analysis or moral observations."
)

SUMMARIZATION_USER = (
    "Summarize the following passage. Focus on: specific events and their outcomes, "
    "characters introduced and what they do, key dialogue or decisions, and any named "
    "locations or objects that matter. Be concise but concrete — prefer 'X kills Y' "
    "over 'themes of violence emerge'.\n\n{context}"
)

# --- Retrieval query used when summarizing a document ---
# Controls which nodes RAPTOR retrieves from the tree.
# More specific queries bias retrieval toward event/detail nodes over abstract theme nodes.

RETRIEVAL_QUERY = "What are the key events, characters, and plot developments in this text?"

# --- Chain of Density (CoD) prompting for the final summary ---
# Based on Adams et al. 2023: "From Sparse to Dense: GPT-4 Summarization with CoD"
# Human-preferred density is at step 3 (entity density ~0.15, matching human-written summaries).
# Applied to the RAPTOR-retrieved context to maximally exploit the rich retrieved content.

COD_SYSTEM = (
    "You are a literary summarization assistant. You extract key narrative elements "
    "from text and write summaries that cover all of them with specific, concrete language. "
    "Summary length is determined by the amount of important content — more events mean "
    "a longer summary. Never replace specific events with abstract thematic statements."
)

COD_USER = """\
Text:
{context}

Step 1 — Extract every key narrative element from the text. A narrative element is:
  - A character and their specific action or decision
  - A specific plot event and its outcome
  - A key revelation, conflict, or named detail that matters to the story

List them as a JSON array of short phrases (5-10 words each). Be exhaustive — include \
every distinct event, not just the most dramatic ones.

Step 2 — Write a narrative summary that covers ALL listed elements. Rules:
  - Use concrete, specific language: prefer "Guppy tells Lady Dedlock the letters are \
destroyed" over "information is shared about correspondence"
  - Order events chronologically where possible
  - One or two sentences per element is typical; length follows from content
  - The summary must be self-contained and readable without the source text
  - Do not add padding, analysis, or thematic observations not grounded in specific events

Return JSON only — no markdown:
{{"elements": ["element 1", "element 2", ...], "summary": "..."}}"""


# --- Final document summary (pipeline output) ---

DOCUMENT_SUMMARY_SYSTEM = (
    "You are a literary summarization assistant. Write summaries that capture "
    "the narrative arc: what happens, who is involved, in what order, and what "
    "the outcome is. Avoid abstract thematic statements. Prefer specific, concrete "
    "language over literary analysis."
)

DOCUMENT_SUMMARY_USER = (
    "Write a concise but complete summary of the document based on the following content. "
    "Include: the main characters and their actions, the key events in sequence, "
    "important decisions or conflicts, and how they resolve. "
    "Do not substitute specific events with vague thematic observations.\n\n{context}"
)

# --- Technical document prompts (IAEA / regulatory / scientific) ---
# Separate from the literary prompts above — focused on requirements, classifications,
# definitions, and regulatory structure rather than narrative events.

TECH_SUMMARIZATION_SYSTEM = (
    "You are a technical document summarizer. Your job is to compress regulatory or "
    "scientific text while preserving: key definitions, classifications, requirements, "
    "recommendations, and scope statements. Use precise technical language. Do not "
    "replace specific requirements with vague thematic statements."
)

TECH_SUMMARIZATION_USER = (
    "Summarize the following passage from a technical document. Focus on: specific "
    "definitions and classifications, key requirements or recommendations (SHALL/SHOULD), "
    "scope and applicability statements, and any named categories or criteria. "
    "Be concise but preserve technical precision.\n\n{context}"
)

TECH_RETRIEVAL_QUERY = (
    "What are the key definitions, classifications, requirements, and recommendations "
    "in this document?"
)

TECH_COD_SYSTEM = (
    "You are a technical summarization assistant. You extract key information elements "
    "from regulatory or scientific documents and write summaries that cover all of them "
    "with precise technical language. Summary length is determined by the amount of "
    "important content. Never replace specific requirements with vague statements."
)

TECH_COD_USER = """\
Text:
{context}

Step 1 — Extract every key information element from the text. An element is:
  - A definition or classification (e.g. "Category A waste: short-lived low-level")
  - A requirement or recommendation (e.g. "SHALL be isolated from the biosphere")
  - A scope statement or applicability condition
  - A named criterion, threshold, or procedure

List them as a JSON array of short phrases (5-10 words each). Be exhaustive.

Step 2 — Write a technical summary that covers ALL listed elements. Rules:
  - Use precise technical language; preserve SHALL/SHOULD/MAY distinctions if present
  - Group related elements logically (definitions → scope → requirements)
  - Length follows from content — more elements mean a longer summary
  - The summary must be self-contained without the source text

Return JSON only — no markdown:
{{"elements": ["element 1", "element 2", ...], "summary": "..."}}"""

# --- Golden standard generation ---
# Uses a strong LLM to generate a reference summary from the full document text.
# This substitutes for human-written references when none exist.

GOLDEN_STANDARD_SYSTEM = (
    "You are an expert technical writer creating a comprehensive reference summary "
    "of a regulatory or scientific document. Your summary will be used as a gold "
    "standard for evaluating other summaries. Be thorough, precise, and cover all "
    "major sections, definitions, requirements, and conclusions."
)

GOLDEN_STANDARD_USER = (
    "Write a comprehensive reference summary of the following document: {title}\n\n"
    "Cover: the document's purpose and scope, key definitions and classifications, "
    "main requirements and recommendations, and any important conclusions or annexes. "
    "This summary will serve as a gold standard — prioritize completeness and accuracy "
    "over brevity.\n\n{text}"
)


# --- LLM-as-judge evaluation ---

JUDGE_SYSTEM = """\
You are an expert evaluator of text summaries. You will be given:
  - SOURCE: an excerpt from the original document
  - REFERENCE: a human-written gold-standard summary
  - SUMMARY A and SUMMARY B: two machine-generated summaries to compare

Score each summary on three dimensions (1–10):
  - Faithfulness: does it avoid hallucinations and stay true to the source?
  - Coverage: does it capture the key points present in the reference?
  - Conciseness: is it informative without unnecessary padding?

Then declare a winner (A, B, or tie) and give a brief justification.

Respond in this exact JSON format (no markdown):
{
  "scores": {
    "A": {"faithfulness": <int>, "coverage": <int>, "conciseness": <int>},
    "B": {"faithfulness": <int>, "coverage": <int>, "conciseness": <int>}
  },
  "winner": "<A|B|tie>",
  "reasoning": "<one or two sentences>"
}"""

JUDGE_USER = """\
SOURCE (excerpt):
{source}

REFERENCE:
{reference}

SUMMARY {label_a}:
{summary_a}

SUMMARY {label_b}:
{summary_b}"""

# --- LLM-as-judge (reference-free) ---
# Used when no gold-standard reference exists (e.g. multi-chapter aggregate texts).
# Coverage is redefined as how well the summary covers the key points in the source.

JUDGE_NO_REF_SYSTEM = """\
You are an expert evaluator of text summaries. You will be given:
  - SOURCE: an excerpt from the original document
  - SUMMARY A and SUMMARY B: two machine-generated summaries to compare

There is no human reference summary. Judge purely based on the source text.

Score each summary on three dimensions (1–10):
  - Faithfulness: does it avoid hallucinations and stay true to the source?
  - Coverage: does it capture the key events, characters, and themes from the source?
  - Conciseness: is it informative without unnecessary padding?

Then declare a winner (A, B, or tie) and give a brief justification.

Respond in this exact JSON format (no markdown):
{
  "scores": {
    "A": {"faithfulness": <int>, "coverage": <int>, "conciseness": <int>},
    "B": {"faithfulness": <int>, "coverage": <int>, "conciseness": <int>}
  },
  "winner": "<A|B|tie>",
  "reasoning": "<one or two sentences>"
}"""

JUDGE_NO_REF_USER = """\
SOURCE (excerpt):
{source}

SUMMARY {label_a}:
{summary_a}

SUMMARY {label_b}:
{summary_b}"""
