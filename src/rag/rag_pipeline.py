"""
rag_pipeline.py

Orchestrates the RAG (Retrieval-Augmented Generation) system: ingesting
PDF annual reports into the vector store, and answering questions about
a company using only its own ingested report text.

Ingestion path: src.rag.document_loader (extract + chunk PDF text) ->
src.rag.embeddings (embed chunks, RETRIEVAL_DOCUMENT task type) ->
src.rag.vector_store (persist to ChromaDB).

Question-answering path: src.rag.embeddings (embed the question,
RETRIEVAL_QUERY task type) -> src.rag.vector_store (similarity search,
scoped to one company) -> src.genai.llm_utils.generate_text() (answer,
grounded only in the retrieved excerpts).

Responsible-use framing: like the Phase 10 AI Research Assistant, this
chatbot is constrained by its system instruction to answer only from the
provided report excerpts (not general knowledge, which could contradict
what the actual filing says), to say clearly when the excerpts don't
contain an answer rather than guessing, and to stay factual/informational
rather than drift into investment advice — a natural risk when a chatbot
is discussing a company's own risk-factors section.

Usage (from project root, with venv activated):
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
    """
    The result of answer_question().

    Attributes:
        symbol: The company the question was about.
        question: The question that was asked.
        answer: Gemini's answer, grounded in the retrieved excerpts.
        sources: Deduplicated (source_file, page) dicts for the excerpts
            that were retrieved, in first-retrieved (most relevant) order.
    """

    symbol: str
    question: str
    answer: str
    sources: list[dict]


def build_rag_prompt(question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
    """
    Build the user-turn prompt for a RAG question: the retrieved excerpts
    (labeled with their page/source) followed by the question itself.

    Args:
        question: The user's question.
        retrieved_chunks: Output of src.rag.vector_store.query(), in
            relevance order.

    Returns:
        A prompt string ready to pass to
        src.genai.llm_utils.generate_text() as `prompt`.
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
    Answer a question about a company using only its ingested report text.

    Args:
        symbol: Ticker symbol whose ingested report(s) to search.
        question: The user's question.
        top_k: How many excerpts to retrieve and include as context.

    Returns:
        None if no report chunks have been ingested for this symbol yet
        (nothing to answer from — the caller should render a "no report
        uploaded yet" state rather than asking Gemini to answer from
        nothing). Otherwise a RAGAnswer.

    Raises:
        LLMConfigError: If GEMINI_API_KEY isn't set.
        LLMRequestError: If either the embedding call or the answer
            generation call fails.
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
    Ingest a single PDF report: extract/chunk its text, embed the
    chunks, and store them in the vector store.

    Args:
        pdf_path: Path to the PDF file.
        symbol: Ticker symbol this report belongs to.

    Returns:
        The number of chunks ingested. 0 if the PDF had no extractable
        text (logged as a warning by document_loader, not raised).

    Raises:
        LLMConfigError: If GEMINI_API_KEY isn't set.
        LLMRequestError: If the embedding call fails.
    """
    chunks = load_report_chunks(pdf_path, symbol)

    if not chunks:
        return 0

    vectors = embed_documents([chunk.text for chunk in chunks])
    return add_chunks(chunks, vectors)


def ingest_all_reports(reports_dir: Path = DEFAULT_REPORTS_DIR) -> dict[str, int]:
    """
    Discover and ingest every recognizably-named PDF in a directory (see
    src.rag.document_loader's module docstring for the filename
    convention).

    Args:
        reports_dir: Directory to scan for .pdf files.

    Returns:
        A dict mapping each symbol to the total number of chunks ingested
        for it (a symbol with multiple report files sums across them).
        Empty dict if no recognizably-named PDFs were found.
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
    """Parse command-line arguments for standalone script execution."""
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
    """Entry point for standalone script execution."""
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
