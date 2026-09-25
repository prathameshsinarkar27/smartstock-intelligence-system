"""
preprocess.py

Cleans news article titles and content before sentiment scoring.

The module handles common issues in syndicated news data, including
NewsAPI truncation markers, HTML fragments, URLs, and inconsistent
whitespace. It also combines the cleaned title and content into the
text passed to the sentiment model.
"""

import re

_TRUNCATION_SUFFIX_RE = re.compile(r"\s*\[\+\d+\s*chars\]\s*$", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+")
_WHITESPACE_RE = re.compile(r"\s+")


def remove_truncation_suffix(text: str) -> str:
    """
    Remove a trailing NewsAPI truncation marker such as "[+1842 chars]".
    """
    return _TRUNCATION_SUFFIX_RE.sub("", text)


def strip_html_tags(text: str) -> str:
    """
    Remove HTML tags while preserving their text content.
    """
    return _HTML_TAG_RE.sub(" ", text)


def strip_urls(text: str) -> str:
    """
    Remove URLs from article text.
    """
    return _URL_RE.sub(" ", text)


def normalize_whitespace(text: str) -> str:
    """
    Collapse consecutive whitespace and trim the result.
    """
    return _WHITESPACE_RE.sub(" ", text).strip()


def clean_text(text: str | None) -> str:
    """
    Apply the complete text-cleaning pipeline to a title or article body.

    Returns an empty string when the input is None or empty.
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
    Combine cleaned title and content into the text used for sentiment scoring.

    The title is retained because it often provides a concise sentiment
    signal, while article content may be truncated.
    """
    clean_title = clean_text(title)
    clean_content = clean_text(content)

    if not clean_content or clean_content == clean_title:
        return clean_title

    if not clean_title:
        return clean_content

    return f"{clean_title}. {clean_content}"
