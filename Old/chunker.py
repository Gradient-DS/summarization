"""
chunker.py
----------
Splits a QASPER document into ~100-token chunks, preserving sentence boundaries.
These chunks form the leaf nodes of the RAPTOR tree.

QASPER document structure:
{
    "title": str,
    "abstract": str,
    "full_text": {
        "section_name": [str, ...],
        "paragraphs": [[str, ...], ...]   # one list of paragraphs per section
    }
}
"""

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    """A single leaf-level text chunk."""
    text: str
    token_count: int
    doc_id: str
    section: str
    chunk_index: int


def simple_tokenize(text: str) -> list[str]:
    """
    Naive whitespace tokenizer used purely for token counting.
    Consistent with how RAPTOR counts tokens (no tiktoken needed at this stage).
    """
    return text.split()


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences using simple punctuation rules.
    Keeps sentences intact so we never cut mid-sentence.
    """
    # Split on '.', '!', '?' followed by whitespace and a capital letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    doc_id: str,
    section: str,
    chunk_size: int = 100,
    start_index: int = 0
) -> list[Chunk]:
    """
    Split a block of text into chunks of approximately `chunk_size` tokens.
    Never cuts mid-sentence: if adding a sentence exceeds the limit,
    the current chunk is saved and a new one starts with that sentence.

    Args:
        text:        The raw text to chunk.
        doc_id:      Identifier for the source document.
        section:     Section name this text came from.
        chunk_size:  Target token count per chunk (default 100, per RAPTOR paper).
        start_index: Starting chunk index (for correct indexing across sections).

    Returns:
        List of Chunk objects.
    """
    sentences = split_into_sentences(text)
    chunks = []
    current_sentences = []
    current_tokens = 0
    chunk_index = start_index

    for sentence in sentences:
        sentence_tokens = len(simple_tokenize(sentence))

        # If a single sentence exceeds chunk_size, it becomes its own chunk
        if sentence_tokens >= chunk_size:
            # Save whatever we have accumulated first
            if current_sentences:
                chunk_text_str = " ".join(current_sentences)
                chunks.append(Chunk(
                    text=chunk_text_str,
                    token_count=current_tokens,
                    doc_id=doc_id,
                    section=section,
                    chunk_index=chunk_index
                ))
                chunk_index += 1
                current_sentences = []
                current_tokens = 0

            # The long sentence becomes its own chunk
            chunks.append(Chunk(
                text=sentence,
                token_count=sentence_tokens,
                doc_id=doc_id,
                section=section,
                chunk_index=chunk_index
            ))
            chunk_index += 1
            continue

        # If adding this sentence exceeds the budget, save the current chunk
        if current_tokens + sentence_tokens > chunk_size and current_sentences:
            chunk_text_str = " ".join(current_sentences)
            chunks.append(Chunk(
                text=chunk_text_str,
                token_count=current_tokens,
                doc_id=doc_id,
                section=section,
                chunk_index=chunk_index
            ))
            chunk_index += 1
            current_sentences = []
            current_tokens = 0

        current_sentences.append(sentence)
        current_tokens += sentence_tokens

    # Save any remaining sentences as a final chunk
    if current_sentences:
        chunk_text_str = " ".join(current_sentences)
        chunks.append(Chunk(
            text=chunk_text_str,
            token_count=current_tokens,
            doc_id=doc_id,
            section=section,
            chunk_index=chunk_index
        ))

    return chunks


def extract_text_from_qasper(document: dict) -> list[tuple[str, str]]:
    """
    Extract (section_name, text) pairs from a QASPER document.
    Includes the title, abstract, and all full-text sections.

    Args:
        document: A single QASPER dataset entry.

    Returns:
        List of (section_name, paragraph_text) tuples, in document order.
    """
    sections = []

    # Title and abstract are always included
    if document.get("title"):
        sections.append(("title", document["title"]))
    if document.get("abstract"):
        sections.append(("abstract", document["abstract"]))

    # Full text: section_name[i] maps to paragraphs[i]
    full_text = document.get("full_text", {})
    section_names = full_text.get("section_name", [])
    paragraphs_per_section = full_text.get("paragraphs", [])

    for section_name, paragraphs in zip(section_names, paragraphs_per_section):
        # Each section has a list of paragraphs — join them
        section_text = " ".join(p for p in paragraphs if p and p.strip())
        if section_text.strip():
            sections.append((section_name or "unknown", section_text))

    return sections


def chunk_qasper_document(
    document: dict,
    doc_id: str,
    chunk_size: int = 100
) -> list[Chunk]:
    """
    Full pipeline: extract text from a QASPER document and chunk it.

    Args:
        document:   A single QASPER dataset entry.
        doc_id:     A unique identifier for this document.
        chunk_size: Target token count per chunk.

    Returns:
        Ordered list of Chunk objects forming the leaf nodes.
    """
    sections = extract_text_from_qasper(document)
    all_chunks = []

    for section_name, section_text in sections:
        section_chunks = chunk_text(
            text=section_text,
            doc_id=doc_id,
            section=section_name,
            chunk_size=chunk_size,
            start_index=len(all_chunks)
        )
        all_chunks.extend(section_chunks)

    return all_chunks


# ---------------------------------------------------------------------------
# Quick sanity check — run this file directly to test on one QASPER document
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from datasets import load_dataset

    print("Loading QASPER dataset...")
    dataset = load_dataset("allenai/qasper", split="train")
    document = dataset[0]

    print(f"Document title: {document['title']}")
    print(f"Sections in full_text: {document['full_text']['section_name'][:5]}")

    chunks = chunk_qasper_document(document, doc_id="qasper_train_0")

    print(f"\nTotal chunks: {len(chunks)}")
    print(f"Avg tokens per chunk: {sum(c.token_count for c in chunks) / len(chunks):.1f}")
    print(f"\nFirst 3 chunks:")
    for chunk in chunks[:3]:
        print(f"  [{chunk.chunk_index}] section='{chunk.section}' "
              f"tokens={chunk.token_count} | {chunk.text[:80]}...")
