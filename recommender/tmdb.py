from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import requests


MediaType = Literal["movie", "tv"]


@dataclass(frozen=True)
class TrailerHit:
    """
    TMDB stores trailers as "videos" which frequently point to YouTube keys.
    `youtube_video_id` is directly embeddable via:
      https://www.youtube-nocookie.com/embed/<id>
    """

    youtube_video_id: str
    tmdb_id: int
    media_type: MediaType
    name: str


class TMDBTrailerResolver:
    """
    Trailer resolution via TMDB (preferred over YouTube Data API).

    Why:
    - avoids YouTube search quota burn
    - TMDB already links titles -> trailers (often YouTube keys)
    - faster and more predictable matching

    Requires:
    - TMDB_API_KEY in env (or passed explicitly)
    """

    def __init__(self, api_key: str | None = None, timeout_s: float = 6.0):
        # TMDB supports:
        # - v3 API key via `api_key` query param
        # - v4 access token via `Authorization: Bearer ...`
        self.api_key = (api_key or os.environ.get("TMDB_API_KEY", "") or "").strip()
        self.access_token = (os.environ.get("TMDB_ACCESS_TOKEN", "") or "").strip()
        self.timeout_s = float(timeout_s)
        self._cache: dict[str, TrailerHit | None] = {}

    def enabled(self) -> bool:
        return bool(self.access_token or self.api_key)

    def resolve(self, *, title: str, year: str | int | None = None, kind: MediaType | None = None) -> TrailerHit | None:
        """
        Best-effort resolution:
        - search in TMDB to find the most likely title
        - fetch /videos for that item
        - pick the best YouTube trailer key
        """
        if not self.enabled():
            return None

        t = (title or "").strip()
        if not t:
            return None

        y = str(year).strip() if year is not None else ""
        k = (kind or "").strip().lower()
        cache_key = f"{t}|{y}|{k}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        hit = self._resolve_uncached(title=t, year=y or None, kind=(k if k in {"movie", "tv"} else None))  # type: ignore[arg-type]
        self._cache[cache_key] = hit
        return hit

    def _resolve_uncached(self, *, title: str, year: str | None, kind: MediaType | None) -> TrailerHit | None:
        picked = self._pick_tmdb_item(title=title, year=year, kind=kind)
        if not picked:
            return None
        tmdb_id, media_type = picked
        return self._pick_trailer_from_videos(tmdb_id=tmdb_id, media_type=media_type)

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any] | None:
        url = f"https://api.themoviedb.org/3{path}"
        headers: dict[str, str] = {}
        qp = dict(params)
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        else:
            qp["api_key"] = self.api_key
        try:
            r = requests.get(
                url,
                params=qp,
                headers=headers or None,
                timeout=self.timeout_s,
            )
        except Exception:
            return None
        if r.status_code != 200:
            return None
        try:
            return r.json() or {}
        except Exception:
            return None

    def _pick_tmdb_item(self, *, title: str, year: str | None, kind: MediaType | None) -> tuple[int, MediaType] | None:
        """
        Use /search/multi and heuristically pick the best match.
        """
        q = title.strip()
        params: dict[str, Any] = {"query": q, "include_adult": "false", "page": 1}
        data = self._get("/search/multi", params=params)
        if not data:
            return None
        results = [x for x in (data.get("results") or []) if isinstance(x, dict)]

        # Filter to movie/tv only, optionally by kind
        cand: list[dict[str, Any]] = []
        for r in results:
            mt = str(r.get("media_type") or "").strip().lower()
            if mt not in {"movie", "tv"}:
                continue
            if kind and mt != kind:
                continue
            cand.append(r)
        if not cand:
            return None

        def score(r: dict[str, Any]) -> float:
            mt = str(r.get("media_type") or "").strip().lower()
            base = float(r.get("popularity") or 0.0)
            base += 5.0 if mt == "movie" else 0.0  # slight preference for movies when ambiguous

            if year:
                date_key = "release_date" if mt == "movie" else "first_air_date"
                d = str(r.get(date_key) or "")
                if d.startswith(str(year)):
                    base += 100.0
            return base

        best = max(cand, key=score)
        tmdb_id = int(best.get("id") or 0)
        mt = str(best.get("media_type") or "movie").strip().lower()
        if tmdb_id <= 0 or mt not in {"movie", "tv"}:
            return None
        return tmdb_id, (mt if mt in {"movie", "tv"} else "movie")  # type: ignore[return-value]

    def _youtube_trailer_from_items(self, items: list[dict[str, Any]], *, tmdb_id: int, media_type: MediaType) -> TrailerHit | None:
        yt = [v for v in items if str(v.get("site") or "").lower() == "youtube" and str(v.get("key") or "").strip()]
        if not yt:
            return None

        def vscore(v: dict[str, Any]) -> tuple[int, int, int]:
            vtype = str(v.get("type") or "")
            official = bool(v.get("official"))
            t_rank = 3 if vtype == "Trailer" else (2 if vtype == "Teaser" else 1)
            o_rank = 1 if official else 0
            s_rank = int(v.get("size") or 0)
            return (t_rank, o_rank, s_rank)

        best = max(yt, key=vscore)
        return TrailerHit(
            youtube_video_id=str(best.get("key") or "").strip(),
            tmdb_id=int(tmdb_id),
            media_type=media_type,
            name=str(best.get("name") or "").strip(),
        )

    def _pick_trailer_from_videos(self, *, tmdb_id: int, media_type: MediaType) -> TrailerHit | None:
        # Prefer en-US; many titles only have trailers under default / all languages.
        for params in ({"language": "en-US"}, {}):
            data = self._get(f"/{media_type}/{int(tmdb_id)}/videos", params=params)
            if not data:
                continue
            items = [x for x in (data.get("results") or []) if isinstance(x, dict)]
            hit = self._youtube_trailer_from_items(items, tmdb_id=tmdb_id, media_type=media_type)
            if hit is not None:
                return hit
        return None


@lru_cache(maxsize=1)
def default_resolver() -> TMDBTrailerResolver:
    return TMDBTrailerResolver()

