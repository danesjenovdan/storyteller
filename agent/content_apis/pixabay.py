from __future__ import annotations

from typing import Any

import requests


FULL_HD_DIMS: set[tuple[int, int]] = {(1920, 1080), (1080, 1920)}


def search_videos(
    query: str, page: int, api_key: str, min_duration: float
) -> list[dict[str, Any]]:
    """Search Pixabay videos and return normalized results."""
    if not api_key:
        return []

    response = requests.get(
        "https://pixabay.com/api/videos/",
        params={
            "key": api_key,
            "q": query,
            "page": page,
            "per_page": 20,
            "safesearch": "true",
        },
        timeout=10,
    )
    response.raise_for_status()

    data = response.json()
    results: list[dict[str, Any]] = []

    for item in data.get("hits", []):
        duration = item.get("duration", 0)
        if duration < min_duration:
            continue

        video_variant = _pick_video_variant(item.get("videos", {}))
        if not video_variant:
            continue

        preview_image = (
            video_variant.get("thumbnail")
            or item.get("userImageURL")
            or video_variant.get("url")
        )

        results.append(
            {
                "id": item.get("id"),
                "image": preview_image,
                "duration": duration,
                "video_url": video_variant.get("url"),
                "width": video_variant.get("width"),
                "height": video_variant.get("height"),
                "user": item.get("user", "Unknown"),
                "url": item.get("pageURL"),
                "source": "pixabay",
            }
        )

    return results


def search_images(
    query: str,
    page: int,
    api_key: str,
    duration: float,
) -> list[dict[str, Any]]:
    """Search Pixabay images and return normalized results."""
    if not api_key:
        return []

    response = requests.get(
        "https://pixabay.com/api/",
        params={
            "key": api_key,
            "q": query,
            "page": page,
            "per_page": 20,
            "orientation": "vertical",
            "safesearch": "true",
        },
        timeout=10,
    )
    response.raise_for_status()

    data = response.json()
    results: list[dict[str, Any]] = []

    for item in data.get("hits", []):
        image_url = item.get("largeImageURL") or item.get("webformatURL")

        results.append(
            {
                "id": item.get("id"),
                "image": item.get("previewURL") or image_url,
                "video_url": image_url,
                "width": item.get("imageWidth"),
                "height": item.get("imageHeight"),
                "user": item.get("user", "Unknown"),
                "url": item.get("pageURL"),
                "is_image": True,
                "duration": duration,
                "source": "pixabay",
            }
        )

    return results


def _pick_video_variant(videos: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Prefer Full HD variant; otherwise return the largest available variant."""
    if not videos:
        return None

    fallback_with_url: dict[str, Any] | None = None
    candidates: list[tuple[bool, int, int, dict[str, Any]]] = []

    for index, key in enumerate(("large", "medium", "small", "tiny")):
        variant = videos.get(key)
        if not variant:
            continue

        url = variant.get("url")
        if not url:
            continue

        if fallback_with_url is None:
            fallback_with_url = variant

        width = variant.get("width")
        height = variant.get("height")
        if not isinstance(width, int) or not isinstance(height, int):
            continue
        if width <= 0 or height <= 0:
            continue

        area = width * height
        is_full_hd = (width, height) in FULL_HD_DIMS
        # Earlier keys win ties to keep deterministic behavior.
        tie_breaker = -index
        candidates.append((is_full_hd, area, tie_breaker, variant))

    if candidates:
        return max(candidates, key=lambda item: (item[0], item[1], item[2]))[3]

    return fallback_with_url
