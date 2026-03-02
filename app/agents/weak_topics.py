"""
Persist and query weak topics over time so the exam assistant can adapt practice and revision.
"""

import json
import os
from collections import defaultdict
from typing import Any


def _weak_topics_path() -> str:
    root = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(root, "data", "processed", "weak_topics.json")


def _load_raw() -> list[dict[str, Any]]:
    path = _weak_topics_path()
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except (OSError, json.JSONDecodeError):
        return []
    return []


def record_weak_topics(topics: list[str], confidence: float) -> None:
    """
    Append weak topics with associated confidence to persistent storage.

    Example entry:
    {"topic": "Dimensional Modeling", "confidence": 0.42}
    """
    if not topics:
        return
    path = _weak_topics_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = _load_raw()
    for topic in topics:
        if topic:
            data.append({"topic": topic, "confidence": float(confidence)})
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError:
        # Best-effort persistence; ignore failures.
        return


def summarize_weak_topics() -> list[dict[str, Any]]:
    """
    Aggregate weak topics over time.

    Returns a list of dicts:
    {
      "topic": str,
      "avg_confidence": float,
      "count": int,
    }
    sorted by ascending avg_confidence (weakest first).
    """
    raw = _load_raw()
    if not raw:
        return []
    scores: dict[str, list[float]] = defaultdict(list)
    for entry in raw:
        topic = str(entry.get("topic", "")).strip()
        if not topic:
            continue
        try:
            conf = float(entry.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        scores[topic].append(conf)
    summary: list[dict[str, Any]] = []
    for topic, vals in scores.items():
        if not vals:
            continue
        avg = sum(vals) / len(vals)
        summary.append(
            {
                "topic": topic,
                "avg_confidence": avg,
                "count": len(vals),
            }
        )
    summary.sort(key=lambda x: x["avg_confidence"])
    return summary


def top_weak_topics(n: int = 5) -> list[str]:
    """
    Return up to n weakest topics by average confidence.
    These can be used to prioritize practice questions or suggest revision topics.
    """
    return [item["topic"] for item in summarize_weak_topics()[:n]]

