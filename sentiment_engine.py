"""
Sentiment and YouTube fame scoring.
 Investor Sentiment criterion C2 (News + YouTube fame).
"""
from __future__ import annotations

from typing import List, Dict, Optional, Tuple
import numpy as np

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception:  # pragma: no cover
    SentimentIntensityAnalyzer = None  # type: ignore


def compute_news_sentiment(articles: List[Dict]) -> Optional[float]:
    if not articles or SentimentIntensityAnalyzer is None:
        return None
    analyzer = SentimentIntensityAnalyzer()
    vals = []
    for a in articles:
        title = (a.get("title") or "") if isinstance(a, dict) else ""
        desc = (a.get("description") or "") if isinstance(a, dict) else ""
        text = f"{title}. {desc}".strip()
        if not text:
            continue
        vals.append(analyzer.polarity_scores(text).get("compound", 0.0))
    if not vals:
        return None
    return float(np.mean(vals))


def _parse_int(x) -> int:
    try:
        if isinstance(x, str):
            return int(x.replace(",", ""))
        return int(x)
    except Exception:
        return 0


def compute_youtube_fame_score(items: List[Dict]) -> Optional[float]:
    """
    Nonlinear min-max scaling on views, likes, comments (log1p + sqrt).
    Returns fame score in [0, 1].
    """
    if not items:
        return None
    views, likes, comments = [], [], []
    for item in items:
        stats = item.get("statistics", {}) if isinstance(item, dict) else {}
        views.append(_parse_int(stats.get("viewCount", 0)))
        likes.append(_parse_int(stats.get("likeCount", 0)))
        comments.append(_parse_int(stats.get("commentCount", 0)))

    def norm_nonlinear(values: List[int]) -> np.ndarray:
        arr = np.log1p(np.array(values, dtype=float))
        mn, mx = float(arr.min()), float(arr.max())
        if mx == mn:
            return np.ones_like(arr) * 0.5
        scaled = (arr - mn) / (mx - mn)
        return np.sqrt(scaled)

    v = norm_nonlinear(views)
    l = norm_nonlinear(likes)
    c = norm_nonlinear(comments)
    fame = np.mean(np.vstack([v, l, c]), axis=0)
    return float(np.mean(fame))


def compute_investor_sentiment_score(news_sent: Optional[float], fame_score: Optional[float], use_news: bool = True, use_youtube: bool = True) -> Optional[float]:
    """Normalize to [0,1] and aggregate for C2."""
    parts = []
    if use_news and news_sent is not None:
        # Map from [-1,1] to [0,1]
        parts.append((news_sent + 1.0) / 2.0)
    if use_youtube and fame_score is not None:
        parts.append(float(np.clip(fame_score, 0.0, 1.0)))
    if not parts:
        return None
    return float(np.mean(parts))
