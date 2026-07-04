"""
sentiment_model.py

VADER sentiment scoring wrapper.

Wraps vaderSentiment's SentimentIntensityAnalyzer behind a small interface
that returns exactly the two values the sentiment_scores table needs
(database/tables.sql, Phase 2): a sentiment label constrained to
('positive', 'negative', 'neutral'), and a confidence_score constrained to
[0, 1]. No other module should import vaderSentiment directly, so the
scoring approach can be swapped later (e.g. a transformer-based classifier)
without changing callers.

VADER (Valence Aware Dictionary and sEntiment Reasoner) was chosen per the
blueprint's tech stack (docs/04_TECH_STACK.md) because it's a lexicon/
rule-based model that requires no training data or GPU, runs fast enough to
score a full news backlog synchronously, and — being tuned on social-media
and short-form text — handles the short headlines and snippets typical of
NewsAPI content reasonably well without fine-tuning.
"""

from dataclasses import dataclass

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.utils.logger import get_logger

logger = get_logger(__name__)

# VADER's own documented thresholds on the compound score for classifying
# a piece of text as positive/negative/neutral.
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05

VALID_SENTIMENT_LABELS = ("positive", "negative", "neutral")


@dataclass(frozen=True)
class SentimentResult:
    """
    Result of scoring one piece of text.

    Attributes:
        sentiment: One of "positive", "negative", "neutral" — matches the
            sentiment_scores.sentiment CHECK constraint exactly.
        confidence_score: The strength of that classification, in [0, 1].
            This is the absolute value of VADER's compound score, not a
            calibrated probability — it measures how strongly polarized
            the text is, which is what "confidence" means for a
            lexicon-based model like VADER.
        compound_score: VADER's raw compound score, in [-1, 1]. Not
            written to the database (the schema only has room for a label
            + a [0, 1] confidence), but exposed here for callers that want
            a signed magnitude, e.g. averaging sentiment across articles
            for a KPI card.
    """

    sentiment: str
    confidence_score: float
    compound_score: float


# A single shared analyzer instance. SentimentIntensityAnalyzer's
# construction loads VADER's lexicon from disk; building one per call
# would repeat that work for every article in a scoring run.
_analyzer = SentimentIntensityAnalyzer()


def classify_sentiment(text: str) -> SentimentResult:
    """
    Score a single piece of text with VADER and classify it into the
    sentiment_scores table's label set.

    Args:
        text: Cleaned text to score, typically the output of
            src.sentiment.preprocess.build_scoring_text().

    Returns:
        A SentimentResult. Empty or whitespace-only text scores as
        neutral with confidence_score 0.0 (VADER's own behavior for text
        with no lexicon hits), rather than raising, so the pipeline can
        still record a row for articles with no usable text.
    """
    if not text or not text.strip():
        return SentimentResult(sentiment="neutral", confidence_score=0.0, compound_score=0.0)

    scores = _analyzer.polarity_scores(text)
    compound = scores["compound"]

    if compound >= POSITIVE_THRESHOLD:
        sentiment = "positive"
    elif compound <= NEGATIVE_THRESHOLD:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    # Clamp defensively: VADER's compound score is documented to fall in
    # [-1, 1], so abs() should already satisfy the table's [0, 1]
    # confidence_score CHECK constraint, but this guards against floating
    # point edge cases (e.g. -1.0000000000000002) causing an insert to
    # fail outright.
    confidence_score = min(1.0, max(0.0, abs(compound)))

    return SentimentResult(
        sentiment=sentiment,
        confidence_score=confidence_score,
        compound_score=compound,
    )
