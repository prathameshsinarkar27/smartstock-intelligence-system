"""
vector_store.py

Wraps ChromaDB for storing and retrieving embedded report chunks.

Uses one collection for all companies and filters retrieval by ticker symbol.
Embeddings are supplied explicitly to preserve the Gemini document/query
embedding configuration.

ChromaDB persists its index under vector_db/.
"""

from dataclasses import dataclass

import chromadb

from src.rag.document_loader import ReportChunk
from src.utils.config import PROJECT_ROOT
from src.utils.logger import get_logger

logger = get_logger(__name__)

VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
COLLECTION_NAME = "annual_reports"

# Cache the client and initialize it lazily.
_client: chromadb.ClientAPI | None = None


@dataclass(frozen=True)
class RetrievedChunk:
    """One chunk returned from a similarity search."""

    text: str
    symbol: str
    source_file: str
    page: int
    distance: float


def _chunk_id(chunk: ReportChunk) -> str:
    """
    Build a deterministic ID for a report chunk.

    Re-ingesting the same chunk updates the existing entry instead of
    creating a duplicate.

    Args:
        chunk: Report chunk to identify.

    Returns:
        ID based on symbol, source file, page, and chunk index.
    """
    return f"{chunk.symbol}::{chunk.source_file}::p{chunk.page}::c{chunk.chunk_index}"


def get_collection() -> chromadb.Collection:
    """
    Return the persistent ChromaDB collection, creating it if needed.

    Returns:
        The collection used for report chunks.
    """
    global _client

    if _client is None:
        VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
        logger.info("Initialized ChromaDB persistent client at %s", VECTOR_DB_DIR)

    return _client.get_or_create_collection(name=COLLECTION_NAME)


def reset_client() -> None:
    """Clear the cached ChromaDB client; used in tests."""
    global _client
    _client = None


def add_chunks(chunks: list[ReportChunk], chunk_embeddings: list[list[float]]) -> int:
    """
    Upsert report chunks and their embeddings into ChromaDB.

    Args:
        chunks: Report chunks to store.
        chunk_embeddings: Embedding vector for each chunk.

    Returns:
        Number of chunks upserted.

    Raises:
        ValueError: If chunk and embedding counts differ.
    """
    if len(chunks) != len(chunk_embeddings):
        raise ValueError(
            f"chunks ({len(chunks)}) and chunk_embeddings ({len(chunk_embeddings)}) must be the same length."
        )

    if not chunks:
        logger.warning("add_chunks: nothing to add.")
        return 0

    collection = get_collection()
    collection.upsert(
        ids=[_chunk_id(chunk) for chunk in chunks],
        embeddings=chunk_embeddings,
        documents=[chunk.text for chunk in chunks],
        metadatas=[
            {"symbol": chunk.symbol, "source_file": chunk.source_file, "page": chunk.page}
            for chunk in chunks
        ],
    )

    logger.info("Upserted %d chunk(s) into the vector store.", len(chunks))
    return len(chunks)


def query(query_embedding: list[float], symbol: str, top_k: int = 5) -> list[RetrievedChunk]:
    """
    Find similar chunks restricted to one company's reports.

    Args:
        query_embedding: Embedded user query.
        symbol: Company ticker symbol.
        top_k: Maximum number of results.

    Returns:
        Retrieved chunks ordered by similarity.
    """
    collection = get_collection()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"symbol": symbol.upper()},
    )

    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    distances = results["distances"][0] if results["distances"] else []

    return [
        RetrievedChunk(
            text=document,
            symbol=metadata["symbol"],
            source_file=metadata["source_file"],
            page=metadata["page"],
            distance=distance,
        )
        for document, metadata, distance in zip(documents, metadatas, distances)
    ]


def delete_symbol(symbol: str) -> None:
    """
    Delete all stored chunks for a company.

    Args:
        symbol: Company ticker symbol.
    """
    collection = get_collection()
    collection.delete(where={"symbol": symbol.upper()})
    logger.info("Deleted all chunks for %s from the vector store.", symbol.upper())


def count_chunks(symbol: str | None = None) -> int:
    """
    Count stored report chunks.

    Args:
        symbol: Optional ticker symbol to count.

    Returns:
        Number of matching chunks.
    """
    collection = get_collection()

    if symbol is None:
        return collection.count()

    # Use get() for symbol filtering because count() has no where parameter.
    result = collection.get(where={"symbol": symbol.upper()}, include=[])
    return len(result["ids"])
