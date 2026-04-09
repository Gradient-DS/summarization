"""
run_iaea.py
-----------
Quick one-shot summarization of an IAEA document using a strong LLM.

Step 1: download PDF → extract text → summarize with gpt-5.4.

Usage:
    python run_iaea.py
"""

import os
import re
import ssl
import sys

import requests
import urllib3
from requests.adapters import HTTPAdapter

_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_root, "lib"))
sys.path.insert(0, os.path.join(_root, "pipelines"))

from dotenv import load_dotenv
load_dotenv()

from iaea_pipeline import extract_pdf_text, generate_golden_standard

DEFAULT_PDF_CACHE = os.path.join(_root, ".pdf_cache")


class _LaxSSLAdapter(HTTPAdapter):
    """HTTPAdapter that accepts legacy TLS handshakes (needed for nucleus.iaea.org)."""

    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        ctx.set_ciphers("DEFAULT:@SECLEVEL=1")
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        kwargs["ssl_context"] = ctx
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        super().init_poolmanager(*args, **kwargs)


def _download_pdf(item: dict, cache_dir: str = DEFAULT_PDF_CACHE) -> str | None:
    """Download a PDF with a relaxed SSL context and cache it locally."""
    url = item.get("download_url")
    title = item.get("title", "untitled")
    source_id = item.get("source_id", "")

    if not url:
        print(f"No download_url for: {title}")
        return None

    os.makedirs(cache_dir, exist_ok=True)
    safe = re.sub(r"[^\w\-]", "_", title)[:60]
    path = os.path.join(cache_dir, f"{safe}_{source_id}.pdf")

    if os.path.exists(path):
        print(f"Using cached PDF: {path}")
        return path

    print(f"Downloading: {title}")
    session = requests.Session()
    session.mount("https://", _LaxSSLAdapter())
    try:
        resp = session.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        with open(path, "wb") as f:
            f.write(resp.content)
        print(f"Saved to: {path}")
        return path
    except Exception as e:
        print(f"Download failed: {e}")
        return None

# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------

ITEM = {
    "source": "iaea",
    "source_id": "58797451",
    "download_url": (
        "https://nucleus.iaea.org/sites/committees/Policy%20Documents/"
        "Complete%20Collections%20of%20Safety%20Standards/"
        "Complete%20collection%20English/"
        "GSG-1%20Classification%20of%20Radioactive%20Waste.pdf"
    ),
    "title": "GSG-1 Classification of Radioactive Waste",
    "filetype": "PDF",
    "year": 2009,
}

MODEL = "gpt-5.4"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def main():
    print(f"Document : {ITEM['title']} ({ITEM['year']})")
    print(f"Model    : {MODEL}")
    print("-" * 60)

    # 1. Download PDF (cached in .pdf_cache/)
    pdf_path = _download_pdf(ITEM)
    if pdf_path is None:
        print("ERROR: PDF download failed.")
        return

    # 2. Extract text
    print("Extracting text from PDF...")
    text = extract_pdf_text(pdf_path)
    if not text.strip():
        print("ERROR: No text extracted from PDF.")
        return
    print(f"Extracted {len(text.split())} words.")

    # 3. Summarize
    print(f"\nGenerating summary with {MODEL}...\n")
    summary = generate_golden_standard(text, title=ITEM["title"], model=MODEL)

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(summary)
    print("=" * 60)


if __name__ == "__main__":
    main()
