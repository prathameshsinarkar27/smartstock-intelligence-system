"""
rag_pipeline.py

Orchestrates the RAG system for ingesting annual reports and answering
company-specific questions using only retrieved report content.

Ingestion: extract/chunk -> embed -> store in ChromaDB.
QA: embed question -> similarity search -> generate a grounded answer.

Usage:
    python -m src.rag.rag_pipeline ingest-all
    python -m src.rag.rag_pipeline ingest --file data/reports/AAPL_2025_10K.pdf --symbol AAPL
    python -m src.rag.rag_pipeline ask --symbol AAPL --question "What are the main risk factors?"
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

from src.genai.llm_utils import generate_text
from src.rag.document_loader import (
    DEFAULT_REPORTS_DIR,
    discover_reports,
    load_report_chunks,
)
from src.rag.embeddings import embed_documents, embed_query
from src.rag.vector_store import RetrievedChunk, add_chunks, query
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_TOP_K = 5

RAG_SYSTEM_INSTRUCTION = """\
You are a financial document assistant. You answer questions about a company using \
ONLY the report excerpts provided below, drawn from that company's own annual report \
or financial filing.

Follow these rules strictly:
1. Base your answer only on the provided excerpts. Do not use outside or general \
   knowledge about the company — the excerpts are the only source of truth here.
2. If the excerpts don't contain enough information to answer the question, say so \
   plainly rather than guessing or filling in gaps.
3. When a fact in your answer comes from a specific excerpt, note which page it came \
   from, e.g. "(page 12)".
4. You are answering a factual question about the report's contents, not evaluating \
   the stock — do not give investment advice or a buy/sell/hold recommendation.
5. Be concise and direct.
"""


@dataclass(frozen=True)
class RAGAnswer:
    """Result of a RAG question with its retrieved source references."""

    symbol: str
    question: str
    answer: str
    sources: list[dict]


def build_rag_prompt(question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
    """
    Build the prompt from retrieved report excerpts and the user question.

    Args:
        question: User's question.
        retrieved_chunks: Relevant chunks returned by vector search.

    Returns:
        Prompt ready for the text generation model.
    """
    lines = ["Report excerpts:", ""]

    for position, chunk in enumerate(retrieved_chunks, start=1):
        lines.append(f"[Excerpt {position} — {chunk.source_file}, page {chunk.page}]")
        lines.append(chunk.text)
        lines.append("")

    lines += [
        f"Question: {question}",
        "",
        "Answer the question now, following the rules in your system instructions.",
    ]

    return "\n".join(lines)


def answer_question(symbol: str, question: str, top_k: int = DEFAULT_TOP_K) -> RAGAnswer | None:
    """
    Answer a company question using only ingested report content.

    Args:
        symbol: Ticker symbol whose reports should be searched.
        question: User's question.
        top_k: Number of excerpts to retrieve.

    Returns:
        RAGAnswer, or None if no report chunks are available.

    Raises:
        LLMConfigError: If Gemini is not configured.
        LLMRequestError: If embedding or generation fails.
    """
    query_embedding = embed_query(question)
    retrieved = query(query_embedding, symbol=symbol, top_k=top_k)

    if not retrieved:
        logger.info("No ingested report chunks found for %s; nothing to answer from.", symbol)
        return None

    prompt = build_rag_prompt(question, retrieved)
    answer_text = generate_text(prompt, RAG_SYSTEM_INSTRUCTION)

    sources = []
    seen = set()
    for chunk in retrieved:
        key = (chunk.source_file, chunk.page)
        if key in seen:
            continue
        seen.add(key)
        sources.append({"source_file": chunk.source_file, "page": chunk.page})

    return RAGAnswer(symbol=symbol.upper(), question=question, answer=answer_text, sources=sources)


def ingest_report(pdf_path: Path, symbol: str) -> int:
    """
    Ingest one PDF by extracting, chunking, embedding, and storing its text.

    Args:
        pdf_path: Path to the PDF file.
        symbol: Ticker symbol associated with the report.

    Returns:
        Number of chunks ingested, or 0 if no text was extracted.

    Raises:
        LLMConfigError: If Gemini is not configured.
        LLMRequestError: If embedding fails.
    """
    chunks = load_report_chunks(pdf_path, symbol)

    if not chunks:
        return 0

    vectors = embed_documents([chunk.text for chunk in chunks])
    return add_chunks(chunks, vectors)


def ingest_all_reports(reports_dir: Path = DEFAULT_REPORTS_DIR) -> dict[str, int]:
    """
    Discover and ingest all PDFs with recognizable ticker filenames.

    Args:
        reports_dir: Directory to scan for PDF files.

    Returns:
        Mapping of ticker symbols to total ingested chunk counts.
    """
    discovered = discover_reports(reports_dir)

    if not discovered:
        logger.warning("No recognizably-named PDF reports found in %s.", reports_dir)
        return {}

    results: dict[str, int] = {}
    for pdf_path, symbol in discovered:
        logger.info("Ingesting %s as %s...", pdf_path.name, symbol)
        count = ingest_report(pdf_path, symbol)
        results[symbol] = results.get(symbol, 0) + count
        logger.info("Ingested %d chunk(s) from %s.", count, pdf_path.name)

    return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest PDF annual reports into the vector store, or ask a question about one."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest a single PDF report.")
    ingest_parser.add_argument("--file", required=True, type=Path, help="Path to the PDF file.")
    ingest_parser.add_argument(
        "--symbol",
        required=True,
        help="Ticker symbol this report belongs to (overrides filename-based inference).",
    )

    ingest_all_parser = subparsers.add_parser(
        "ingest-all", help="Discover and ingest every recognizably-named PDF in data/reports/."
    )
    ingest_all_parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help=f"Directory to scan for PDFs (default: {DEFAULT_REPORTS_DIR}).",
    )

    ask_parser = subparsers.add_parser("ask", help="Ask a question about a company's ingested report(s).")
    ask_parser.add_argument("--symbol", required=True, help="Ticker symbol to ask about.")
    ask_parser.add_argument("--question", required=True, help="The question to ask.")
    ask_parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K, help=f"Number of excerpts to retrieve (default: {DEFAULT_TOP_K})."
    )

    return parser.parse_args()


def main() -> None:
    """Run the selected RAG command."""
    args = parse_args()

    if args.command == "ingest":
        count = ingest_report(args.file, args.symbol)
        logger.info("Ingested %d chunk(s) from %s for %s.", count, args.file.name, args.symbol.upper())

    elif args.command == "ingest-all":
        results = ingest_all_reports(args.reports_dir)
        if not results:
            logger.info("Nothing ingested.")
        for symbol, count in results.items():
            logger.info("%s: %d chunk(s) ingested.", symbol, count)

    elif args.command == "ask":
        result = answer_question(args.symbol, args.question, top_k=args.top_k)
        if result is None:
            print(f"No ingested report found for {args.symbol.upper()}. Run `ingest` or `ingest-all` first.")
        else:
            print(f"\n{result.answer}\n")
            print("Sources:")
            for source in result.sources:
                print(f"  - {source['source_file']}, page {source['page']}")


if __name__ == "__main__":
    main()
