"""
document_loader.py

Loads PDF annual reports from data/reports/ and splits extracted text
into overlapping chunks for embedding and vector-store ingestion.

Ticker symbols are inferred from the filename prefix before "_", "-", or "."
(e.g. AAPL_2025_10K.pdf -> AAPL).

Handles text extraction and chunking only; embeddings and vector-store
operations are handled by other modules.
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

# Chunk size preserves paragraph context while remaining suitable for embeddings.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# Match 1-5 letter ticker symbols followed by a filename separator.
_SYMBOL_PREFIX_RE = re.compile(r"^([A-Za-z]{1,5})[_\-.]")


@dataclass(frozen=True)
class ReportChunk:
    """One extracted report chunk ready for embedding."""

    symbol: str
    source_file: str
    page: int
    chunk_index: int
    text: str


def infer_symbol_from_filename(filename: str) -> str | None:
    """
    Infer a ticker symbol from a report filename.

    Args:
        filename: PDF filename, such as "AAPL_2025_10K.pdf".

    Returns:
        Uppercase ticker symbol, or None if the filename is invalid.
    """
    match = _SYMBOL_PREFIX_RE.match(filename)
    if match is None:
        return None
    return match.group(1).upper()


def discover_reports(reports_dir: Path = DEFAULT_REPORTS_DIR) -> list[tuple[Path, str]]:
    """
    Find PDFs with recognizable ticker symbols in their filenames.

    Args:
        reports_dir: Directory to scan for PDF files.

    Returns:
        Sorted list of (pdf_path, symbol) tuples.
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
    Extract text from every PDF page.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Page text in document order. Pages with extraction errors return
        an empty string so other pages can still be processed.
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
    Extract and chunk a PDF report.

    Args:
        pdf_path: Path to the PDF file.
        symbol: Ticker symbol associated with the report.

    Returns:
        Report chunks in page and chunk order. Returns an empty list when
        no extractable text is found.
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
