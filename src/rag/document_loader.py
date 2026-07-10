"""
document_loader.py

Loads PDF annual reports/financial documents from data/reports/ (a
directory reserved for this purpose since Phase 0's folder structure —
see .gitignore, which excludes everything in it except .gitkeep, since
these are user-supplied files, not something this repo ships) and splits
each one into overlapping text chunks ready for embedding
(src/rag/embeddings.py) and storage (src/rag/vector_store.py).

File naming convention: a report's ticker symbol is inferred from its
filename's prefix up to the first "_" or "-" or "." character, e.g.:
    AAPL_2025_10K.pdf        -> "AAPL"
    MSFT-annual-report.pdf   -> "MSFT"
    JPM.pdf                  -> "JPM"
This keeps ingestion a single `python -m src.rag.rag_pipeline ingest-all`
command with no extra bookkeeping file — you just name the PDF sensibly.
If a filename doesn't parse to a plausible ticker, that file is skipped
with a logged warning rather than guessed at.

Pure text extraction and chunking — no embedding calls, no database or
vector store access.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src.utils.config import PROJECT_ROOT
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_REPORTS_DIR = PROJECT_ROOT / "data" / "reports"

# Chosen for financial-document text: large enough to keep a paragraph's
# context together (important for numbers that reference an antecedent
# sentence, e.g. "This represents a 12% increase"), with enough overlap
# that a fact split across a chunk boundary is very unlikely to lose the
# entire sentence it belongs to.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# A ticker symbol is 1-5 uppercase letters (this project doesn't track
# any symbols with digits or longer suffixes — see config/tracked_symbols.txt).
_SYMBOL_PREFIX_RE = re.compile(r"^([A-Za-z]{1,5})[_\-.]")


@dataclass(frozen=True)
class ReportChunk:
    """
    One chunk of extracted, split report text, ready for embedding.

    Attributes:
        symbol: The ticker symbol this report belongs to (uppercase).
        source_file: The PDF's filename (not full path — kept short for
            use as part of a vector store document ID).
        page: The 1-indexed page number this chunk's text was extracted
            from. A chunk that spans a page boundary (due to overlap) is
            attributed to the page its first character came from.
        chunk_index: This chunk's position within its source page,
            0-indexed. Combined with symbol/source_file/page, gives a
            stable, deterministic ID for upserting into the vector store.
        text: The chunk's text content.
    """

    symbol: str
    source_file: str
    page: int
    chunk_index: int
    text: str


def infer_symbol_from_filename(filename: str) -> str | None:
    """
    Infer a ticker symbol from a report's filename (see module docstring
    for the naming convention).

    Args:
        filename: A PDF filename, e.g. "AAPL_2025_10K.pdf".

    Returns:
        The inferred uppercase symbol, or None if the filename doesn't
        start with a plausible 1-5 letter ticker followed by a separator.
    """
    match = _SYMBOL_PREFIX_RE.match(filename)
    if match is None:
        return None
    return match.group(1).upper()


def discover_reports(reports_dir: Path = DEFAULT_REPORTS_DIR) -> list[tuple[Path, str]]:
    """
    Find every PDF in `reports_dir` with a filename that parses to a
    ticker symbol.

    Args:
        reports_dir: Directory to scan for .pdf files (non-recursive).

    Returns:
        A list of (pdf_path, symbol) tuples, sorted by filename. PDFs
        whose filename doesn't parse to a symbol are skipped with a
        logged warning, not raised — one badly-named file shouldn't stop
        the rest of the directory from being ingested.
    """
    if not reports_dir.exists():
        logger.warning("Reports directory does not exist: %s", reports_dir)
        return []

    discovered = []
    for pdf_path in sorted(reports_dir.glob("*.pdf")):
        symbol = infer_symbol_from_filename(pdf_path.name)
        if symbol is None:
            logger.warning(
                "Skipping %s: filename doesn't start with a recognizable TICKER_ prefix.",
                pdf_path.name,
            )
            continue
        discovered.append((pdf_path, symbol))

    return discovered


def _extract_pages(pdf_path: Path) -> list[str]:
    """
    Extract raw text from every page of a PDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A list of strings, one per page, in page order. A page pypdf
        can't extract text from (e.g. a scanned image with no text
        layer) contributes an empty string rather than raising, so one
        unreadable page doesn't block the rest of the document.
    """
    reader = PdfReader(pdf_path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            pages.append(page.extract_text() or "")
        except Exception as exc:
            logger.warning("Could not extract text from %s page %d: %s", pdf_path.name, page_number, exc)
            pages.append("")

    return pages


def load_report_chunks(pdf_path: Path, symbol: str) -> list[ReportChunk]:
    """
    Extract and chunk a single PDF report's text.

    Args:
        pdf_path: Path to the PDF file.
        symbol: The ticker symbol this report belongs to (typically from
            discover_reports(), but can be passed explicitly to override
            filename-based inference).

    Returns:
        A list of ReportChunk, in page then chunk_index order. Empty list
        if the PDF has no extractable text at all (e.g. entirely scanned
        images) — logged as a warning, not raised, since a caller
        ingesting many reports shouldn't have one bad file stop the rest.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    pages = _extract_pages(pdf_path)

    chunks = []
    for page_number, page_text in enumerate(pages, start=1):
        if not page_text.strip():
            continue
        for chunk_index, chunk_text in enumerate(splitter.split_text(page_text)):
            chunks.append(
                ReportChunk(
                    symbol=symbol.upper(),
                    source_file=pdf_path.name,
                    page=page_number,
                    chunk_index=chunk_index,
                    text=chunk_text,
                )
            )

    if not chunks:
        logger.warning("No extractable text found in %s — is it a scanned/image-only PDF?", pdf_path.name)

    return chunks
