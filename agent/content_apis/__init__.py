from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from django.conf import settings

from . import pexels, pixabay

DEFAULT_CONTENT_SOURCES: tuple[str, ...] = ("pexels", "pixabay")


_VIDEO_SEARCH_HANDLERS = {
    "pexels": pexels.search_videos,
    "pixabay": pixabay.search_videos,
}

_IMAGE_SEARCH_HANDLERS = {
    "pexels": pexels.search_images,
    "pixabay": pixabay.search_images,
}


def resolve_sources(raw_sources: str | Iterable[str] | None) -> list[str]:
    """Normalize source inputs to known provider names in stable order."""
    normalized: list[str] = []

    if raw_sources is None:
        return list(DEFAULT_CONTENT_SOURCES)

    candidates: list[str] = []
    if isinstance(raw_sources, str):
        candidates = raw_sources.split(",")
    else:
        for item in raw_sources:
            candidates.extend(str(item).split(","))

    for candidate in candidates:
        source = candidate.strip().lower()
        if source in _VIDEO_SEARCH_HANDLERS and source not in normalized:
            normalized.append(source)

    return normalized or list(DEFAULT_CONTENT_SOURCES)


def search_videos_by_sources(
    *,
    sources: list[str],
    query: str,
    page: int,
    min_duration: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Aggregate normalized videos for selected sources."""
    api_keys = {
        "pexels": settings.PEXELS_API_KEY,
        "pixabay": settings.PIXABAY_API_KEY,
    }

    results: list[dict[str, Any]] = []
    warnings: list[str] = []

    for source in sources:
        handler = _VIDEO_SEARCH_HANDLERS.get(source)
        if not handler:
            continue

        api_key = api_keys.get(source, "")
        if not api_key:
            warnings.append(f"{source.upper()} API key is not configured")
            continue

        try:
            results.extend(
                handler(
                    query=query, page=page, api_key=api_key, min_duration=min_duration
                )
            )
        except Exception as exc:
            warnings.append(f"{source.upper()} search failed: {exc}")

    return results, warnings


def search_images_by_sources(
    *,
    sources: list[str],
    query: str,
    page: int,
    duration: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Aggregate normalized images for selected sources."""
    api_keys = {
        "pexels": settings.PEXELS_API_KEY,
        "pixabay": settings.PIXABAY_API_KEY,
    }

    results: list[dict[str, Any]] = []
    warnings: list[str] = []

    for source in sources:
        handler = _IMAGE_SEARCH_HANDLERS.get(source)
        if not handler:
            continue

        api_key = api_keys.get(source, "")
        if not api_key:
            warnings.append(f"{source.upper()} API key is not configured")
            continue

        try:
            results.extend(
                handler(query=query, page=page, api_key=api_key, duration=duration)
            )
        except Exception as exc:
            warnings.append(f"{source.upper()} search failed: {exc}")

    return results, warnings
