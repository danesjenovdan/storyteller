from __future__ import annotations

from typing import Any

import requests


FULL_HD_DIMS: set[tuple[int, int]] = {(1920, 1080), (1080, 1920)}


def search_videos(
    query: str, page: int, api_key: str, min_duration: float
) -> list[dict[str, Any]]:
    """Search Pexels videos and return normalized results."""
    if not api_key:
        return []

    url = "https://api.pexels.com/videos/search"
    headers = {"Authorization": api_key}

    keywords = [*query.split(","), query]
    results: list[dict[str, Any]] = []
    seen_ids: set[int] = set()

    for keyword in keywords:
        keyword = keyword.strip()
        if not keyword:
            continue

        params = {
            "query": keyword,
            "per_page": 10,
            "page": page,
            "size": "medium",
        }

        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        for item in data.get("videos", []):
            video_duration = item.get("duration", 0)
            if video_duration < min_duration:
                continue

            video_file = _pick_preferred_video_file(item.get("video_files", []))
            if not video_file:
                continue

            video_id = item.get("id")
            if video_id in seen_ids:
                continue
            seen_ids.add(video_id)

            results.append(
                {
                    "id": video_id,
                    "image": item.get("image"),
                    "duration": video_duration,
                    "video_url": video_file.get("link"),
                    "width": video_file.get("width"),
                    "height": video_file.get("height"),
                    "user": item.get("user", {}).get("name", "Unknown"),
                    "url": item.get("url"),
                    "source": "pexels",
                }
            )

    return results


def search_images(
    query: str,
    page: int,
    api_key: str,
    duration: float,
) -> list[dict[str, Any]]:
    """Search Pexels images and return normalized results."""
    if not api_key:
        return []

    headers = {"Authorization": api_key}
    response = requests.get(
        "https://api.pexels.com/v1/search",
        headers=headers,
        params={
            "query": query,
            "per_page": 20,
            "page": page,
            "orientation": "portrait",
        },
        timeout=10,
    )
    response.raise_for_status()

    data = response.json()
    results: list[dict[str, Any]] = []

    for photo in data.get("photos", []):
        src = photo.get("src", {})
        image_url = src.get("large") or src.get("large2x") or src.get("original")

        results.append(
            {
                "id": photo.get("id"),
                "image": src.get("medium"),
                "video_url": image_url,
                "width": photo.get("width"),
                "height": photo.get("height"),
                "user": photo.get("photographer", "Unknown"),
                "url": photo.get("url"),
                "is_image": True,
                "duration": duration,
                "source": "pexels",
            }
        )

    return results


def _pick_preferred_video_file(
    video_files: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Prefer Full HD video; otherwise return the largest available file."""
    if not video_files:
        return None

    fallback_with_url: dict[str, Any] | None = None
    candidates: list[tuple[bool, int, dict[str, Any]]] = []

    for file_obj in video_files:
        link = file_obj.get("link")
        if not link:
            continue

        if fallback_with_url is None:
            fallback_with_url = file_obj

        width = file_obj.get("width")
        height = file_obj.get("height")
        if not isinstance(width, int) or not isinstance(height, int):
            continue
        if width <= 0 or height <= 0:
            continue

        area = width * height
        is_full_hd = (width, height) in FULL_HD_DIMS
        candidates.append((is_full_hd, area, file_obj))

    if candidates:
        return max(candidates, key=lambda item: (item[0], item[1]))[2]

    return fallback_with_url
