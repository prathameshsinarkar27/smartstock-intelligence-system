"""
preprocess.py

Text cleaning for news article titles/content, ahead of sentiment scoring
(src/sentiment/sentiment_model.py).

This module is pure text transformation: it takes raw strings pulled from
the news_articles table (as fetched by src/ingestion/fetch_news.py and
loaded by src/etl/load_to_db.py) and returns cleaned strings.

Two known NewsAPI.org quirks this module specifically corrects for:
    1. The free tier truncates `content` and appends a marker like
       "... [+1234 chars]" — left in place, that marker's digits would be
       fed to the sentiment model as ordinary text.
    2. `content`/`description` fields occasionally contain raw HTML
       fragments (e.g. an <a> tag) from source sites that don't fully
       strip markup before syndication.
"""

import re

_TRUNCATION_SUFFIX_RE = re.compile(r"\s*\[\+\d+\s*chars\]\s*$", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+")
_WHITESPACE_RE = re.compile(r"\s+")


def remove_truncation_suffix(text: str) -> str:
    """
    Strip NewsAPI.org's free-tier truncation marker from the end of a
    string, e.g. "Apple posted record revenue... [+1842 chars]" ->
    "Apple posted record revenue...".

    Args:
        text: Raw article text, possibly ending in a truncation marker.

    Returns:
        The text with any trailing "[+N chars]" marker removed. Text
        without the marker is returned unchanged.
    """
    return _TRUNCATION_SUFFIX_RE.sub("", text)


def strip_html_tags(text: str) -> str:
    """
    Remove HTML tags from a string, leaving the tag's text content intact.

    This is a lightweight regex strip, not a full HTML parser — sufficient
    for the occasional stray tag in syndicated news content, not intended
    for general-purpose HTML sanitization.

    Args:
        text: Raw text, possibly containing HTML tags.

    Returns:
        The text with tags removed (contents preserved).
    """
    return _HTML_TAG_RE.sub(" ", text)


def strip_urls(text: str) -> str:
    """
    Remove raw URLs from a string. URLs carry no sentiment signal and can
    otherwise dominate a short article snippet's word count.

    Args:
        text: Raw text, possibly containing URLs.

    Returns:
        The text with URLs removed.
    """
    return _URL_RE.sub(" ", text)


def normalize_whitespace(text: str) -> str:
    """
    Collapse runs of whitespace (including newlines/tabs left over from
    tag/URL removal) into single spaces, and trim the result.

    Args:
        text: Raw text.

    Returns:
        The text with whitespace normalized.
    """
    return _WHITESPACE_RE.sub(" ", text).strip()


def clean_text(text: str | None) -> str:
    """
    Apply the full cleaning pipeline to a single piece of article text:
    remove the NewsAPI truncation marker, strip HTML tags, strip URLs,
    then normalize whitespace.

    Args:
        text: Raw text (a title or a content/description field). May be
            None or empty, matching how these columns can be stored.

    Returns:
        The cleaned text, or an empty string if `text` was None/empty.
    """
    if not text:
        return ""

    cleaned = remove_truncation_suffix(text)
    cleaned = strip_html_tags(cleaned)
    cleaned = strip_urls(cleaned)
    cleaned = normalize_whitespace(cleaned)
    return cleaned


def build_scoring_text(title: str | None, content: str | None) -> str:
    """
    Combine a cleaned title and cleaned content into the single string
    handed to the sentiment model.

    The title is included even when content is present (rather than
    scoring content alone) because headlines often carry the article's
    strongest, most concise sentiment signal, and NewsAPI content is
    frequently truncated to a short snippet that may lack it entirely.

    Args:
        title: Raw article title.
        content: Raw article content/description (may be shorter than the
            full article — see the NewsAPI truncation note above).

    Returns:
        A single cleaned string: the cleaned title, followed by the
        cleaned content if non-empty and different from the title.
        Returns an empty string if both inputs are empty after cleaning.
    """
    clean_title = clean_text(title)
    clean_content = clean_text(content)

    if not clean_content or clean_content == clean_title:
        return clean_title

    if not clean_title:
        return clean_content

    return f"{clean_title}. {clean_content}"
