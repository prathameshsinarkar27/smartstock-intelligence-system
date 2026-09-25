"""
sentiment_model.py

VADER-based sentiment scoring for news articles.

Provides a small interface that converts text into a sentiment label,
confidence score, and raw compound score. Keeping VADER behind this
module allows the underlying scoring approach to be replaced later
without changing its callers.
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
        sentiment: "positive", "negative", or "neutral".
        confidence_score: Absolute compound-score magnitude in [0, 1].
            This represents the strength of the sentiment signal, not a
            calibrated probability.
        compound_score: Raw VADER compound score in [-1, 1].
    """

    sentiment: str
    confidence_score: float
    compound_score: float


# Reuse one analyzer instance because constructing the analyzer loads
# VADER's lexicon.
_analyzer = SentimentIntensityAnalyzer()


def classify_sentiment(text: str) -> SentimentResult:
    """
    Score text with VADER and classify it as positive, negative, or neutral.

    Empty or whitespace-only text is treated as neutral with zero
    confidence rather than raising an error.
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

    # VADER's compound score is expected to be in [-1, 1]. Clamp the
    # absolute value defensively so confidence always remains in [0, 1].
    confidence_score = min(1.0, max(0.0, abs(compound)))

    return SentimentResult(
        sentiment=sentiment,
        confidence_score=confidence_score,
        compound_score=compound,
    )
