"""
sentiment_pipeline.py

Cleans and scores news articles with VADER, then upserts the results into
the sentiment_scores table.


Usage:
    python -m src.sentiment.sentiment_pipeline
    python -m src.sentiment.sentiment_pipeline --rescore-all
    python -m src.sentiment.sentiment_pipeline --symbols AAPL MSFT
"""

import argparse
from typing import Any

from psycopg2.extras import execute_values

from src.sentiment.preprocess import build_scoring_text
from src.sentiment.sentiment_model import classify_sentiment
from src.utils.database import get_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)

_UNSCORED_QUERY = """
    SELECT na.news_id, na.title, na.content
    FROM news_articles na
    LEFT JOIN sentiment_scores ss ON ss.news_id = na.news_id
    JOIN companies c ON c.company_id = na.company_id
    WHERE ss.score_id IS NULL
      AND (%(symbols)s::text[] IS NULL OR c.symbol = ANY(%(symbols)s));
"""

_ALL_ARTICLES_QUERY = """
    SELECT na.news_id, na.title, na.content
    FROM news_articles na
    JOIN companies c ON c.company_id = na.company_id
    WHERE (%(symbols)s::text[] IS NULL OR c.symbol = ANY(%(symbols)s));
"""

_UPSERT_QUERY = """
    INSERT INTO sentiment_scores (news_id, sentiment, confidence_score)
    VALUES %s
    ON CONFLICT (news_id) DO UPDATE SET
        sentiment = EXCLUDED.sentiment,
        confidence_score = EXCLUDED.confidence_score,
        created_at = now();
"""


def get_articles_to_score(
    symbols: list[str] | None = None,
    rescore_all: bool = False,
) -> list[dict[str, Any]]:
    """Fetch news articles that require sentiment scoring."""
    upper_symbols = [s.upper() for s in symbols] if symbols else None
    query = _ALL_ARTICLES_QUERY if rescore_all else _UNSCORED_QUERY

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, {"symbols": upper_symbols})
            rows = cur.fetchall()

    return [{"news_id": r[0], "title": r[1], "content": r[2]} for r in rows]


def score_articles(articles: list[dict[str, Any]]) -> list[tuple[int, str, float]]:
    """Clean and classify a batch of news articles."""
    results: list[tuple[int, str, float]] = []

    for article in articles:
        text = build_scoring_text(article["title"], article["content"])
        result = classify_sentiment(text)
        results.append((article["news_id"], result.sentiment, result.confidence_score))

    return results


def write_sentiment_scores(scored: list[tuple[int, str, float]]) -> int:
    """Upsert sentiment results into the database."""
    if not scored:
        logger.warning("write_sentiment_scores: nothing to write.")
        return 0

    with get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, _UPSERT_QUERY, scored)

    logger.info("Upserted %d sentiment score row(s).", len(scored))
    return len(scored)


def run_sentiment_pipeline(
    symbols: list[str] | None = None,
    rescore_all: bool = False,
) -> int:
    """Fetch, score, and persist news sentiment results."""
    articles = get_articles_to_score(symbols=symbols, rescore_all=rescore_all)

    if not articles:
        logger.info("No articles to score. Nothing to do.")
        return 0

    logger.info("Scoring %d article(s)...", len(articles))
    scored = score_articles(articles)
    return write_sentiment_scores(scored)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Score news article sentiment with VADER and write results to sentiment_scores."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Restrict scoring to these ticker symbols, e.g. --symbols AAPL MSFT. "
        "Default: all companies with loaded news.",
    )
    parser.add_argument(
        "--rescore-all",
        action="store_true",
        help="Re-score every matching article, including ones that already have a "
        "sentiment_scores row (e.g. after changing preprocess.py or sentiment_model.py).",
    )
    return parser.parse_args()


def main() -> None:
    """Run the sentiment pipeline."""
    args = parse_args()
    written = run_sentiment_pipeline(symbols=args.symbols, rescore_all=args.rescore_all)
    logger.info("Sentiment pipeline complete. %d row(s) written.", written)


if __name__ == "__main__":
    main()
