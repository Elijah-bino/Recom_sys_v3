from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal
from urllib.parse import quote_plus

import requests

from .secrets_io import read_env_style_key_file
from .tmdb import TMDBTrailerResolver

MediaType = Literal["movie", "tv"]

_YT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,}$")
# Match common YouTube URL shapes so UIs can embed even when only a URL is present.
_YT_URL_EXTRACT_RE = re.compile(
    r"(?:youtube\.com/embed/|youtube\.com/shorts/|youtu\.be/|[?&]v=)([A-Za-z0-9_-]{6,})"
)


def extract_video_id_from_url(url: str) -> str:
    """Return the video id if `url` looks like a YouTube watch/embed/short/share link."""
    if not url:
        return ""
    s = str(url).strip()
    m = _YT_URL_EXTRACT_RE.search(s)
    return m.group(1) if m else ""


@dataclass(frozen=True)
class TrailerHit:
    video_id: str
    title: str


class YouTubeTrailerResolver:
    """
    Best-effort trailer lookup using YouTube Data API v3 search.

    Requires YOUTUBE_API_KEY in the environment for real lookups.
    """

    def __init__(self, api_key: str | None = None, timeout_s: float = 5.0):
        self.api_key = (api_key or os.environ.get("YOUTUBE_API_KEY", "") or "").strip()
        key_file = (os.environ.get("YOUTUBE_API_KEY_FILE", "") or "").strip()
        if (not self.api_key) and key_file:
            self.api_key = read_env_style_key_file(key_file, "YOUTUBE_API_KEY").strip()
        self.timeout_s = float(timeout_s)
        self._cache: dict[str, TrailerHit | None] = {}
        self._embeddable_cache: dict[str, bool] = {}
        # Set True after a 403 quotaExceeded so /health can explain missing trailers.
        self.quota_exceeded: bool = False

    def enabled(self) -> bool:
        return bool(self.api_key)

    def resolve_via_tmdb(
        self, *, movie_title: str, year: str | None = None, kind: MediaType | None = None
    ) -> TrailerHit | None:
        """
        Preferred path: use TMDB to find the YouTube trailer key.
        This avoids YouTube search quota usage entirely.
        """
        tmdb = TMDBTrailerResolver()
        if not tmdb.enabled():
            return None
        hit = tmdb.resolve(title=movie_title, year=year, kind=kind)
        if not hit or not hit.youtube_video_id:
            return None
        return TrailerHit(video_id=hit.youtube_video_id, title=hit.name or movie_title)

    def watch_url(self, video_id: str) -> str:
        return f"https://www.youtube.com/watch?v={video_id}"

    def search_url_fallback(self, query: str) -> str:
        return f"https://www.youtube.com/results?search_query={quote_plus(query)}"

    def resolve(self, *, movie_title: str, year: str | None = None, kind: MediaType | None = None) -> TrailerHit | None:
        # 1) Try TMDB first (doesn't consume YouTube Data API quota)
        tmdb_hit = self.resolve_via_tmdb(movie_title=movie_title, year=year, kind=kind)
        if tmdb_hit is not None:
            return tmdb_hit
        # Wrong movie vs tv is common for ambiguous titles — try unrestricted search once.
        if kind is not None:
            tmdb_hit = self.resolve_via_tmdb(movie_title=movie_title, year=year, kind=None)
            if tmdb_hit is not None:
                return tmdb_hit

        # 2) Fallback: YouTube Data API search (quota heavy)
        if not self.enabled():
            return None

        title = (movie_title or "").strip()
        if not title:
            return None

        cache_key = f"{title}|{year or ''}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        q = f"{title} official trailer"
        if year:
            y = str(year).strip()
            if y:
                q = f"{title} {y} official trailer"

        hit = self._search(q)
        self._cache[cache_key] = hit
        return hit

    def _note_quota_from_response(self, r: requests.Response) -> None:
        if r.status_code != 403:
            return
        try:
            data = r.json()
            err = data.get("error") or {}
            for e in err.get("errors") or []:
                if str(e.get("reason") or "") == "quotaExceeded":
                    self.quota_exceeded = True
                    return
            if "quota" in str(err.get("message") or "").lower():
                self.quota_exceeded = True
        except Exception:
            pass

    def _parse_search_hits(self, data: dict[str, Any]) -> list[TrailerHit]:
        candidates: list[TrailerHit] = []
        for it in data.get("items") or []:
            vid = (((it or {}).get("id") or {}).get("videoId")) or ""
            if not isinstance(vid, str) or not _YT_ID_RE.match(vid):
                continue
            sn = it.get("snippet") or {}
            t = str(sn.get("title") or "").strip()
            candidates.append(TrailerHit(video_id=vid, title=t))
        return candidates

    def _run_search(self, base: dict[str, Any], extra: dict[str, Any]) -> list[TrailerHit]:
        """search.list — 100 quota units per call."""
        url = "https://www.googleapis.com/youtube/v3/search"
        params = {**base, **extra}
        try:
            r = requests.get(url, params=params, timeout=self.timeout_s)
        except Exception:
            return []
        if r.status_code != 200:
            self._note_quota_from_response(r)
            return []
        self.quota_exceeded = False
        return self._parse_search_hits(r.json() or {})

    def _videos_embeddable_batch(self, video_ids: list[str]) -> dict[str, bool]:
        """
        One videos.list call for up to 50 ids (1 quota unit) instead of N separate calls.
        """
        ids = [v.strip() for v in video_ids if _YT_ID_RE.match((v or "").strip())]
        if not ids:
            return {}
        ids = ids[:50]
        missing = [i for i in ids if i not in self._embeddable_cache]
        if missing:
            vurl = "https://www.googleapis.com/youtube/v3/videos"
            params: dict[str, Any] = {
                "part": "status",
                "id": ",".join(missing),
                "key": self.api_key,
            }
            try:
                r = requests.get(vurl, params=params, timeout=self.timeout_s)
                if r.status_code != 200:
                    self._note_quota_from_response(r)
                    for i in missing:
                        self._embeddable_cache.setdefault(i, False)
                else:
                    self.quota_exceeded = False
                    seen: set[str] = set()
                    for item in (r.json() or {}).get("items") or []:
                        vid = str(item.get("id") or "").strip()
                        if not vid:
                            continue
                        seen.add(vid)
                        st = item.get("status") or {}
                        self._embeddable_cache[vid] = bool(st.get("embeddable"))
                    for i in missing:
                        if i not in seen:
                            self._embeddable_cache[i] = False
            except Exception:
                for i in missing:
                    self._embeddable_cache.setdefault(i, False)

        return {i: bool(self._embeddable_cache.get(i, False)) for i in ids}

    def _search(self, q: str) -> TrailerHit | None:
        # https://developers.google.com/youtube/v3/docs/search/list
        strict_embed = (os.environ.get("YOUTUBE_REQUIRE_EMBEDDABLE", "") or "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        base: dict[str, Any] = {
            "part": "snippet",
            "type": "video",
            "maxResults": 10,
            "q": q,
            "key": self.api_key,
        }

        # 1) Prefer API-side embeddable filter — avoids up to 10× videos.list calls per title.
        embed_only = self._run_search(base, {"videoEmbeddable": "true"})
        if embed_only:
            return embed_only[0]

        # 2) Broad search, then pick embeddable in one batched videos.list (or top hit if not strict).
        broad = self._run_search(base, {})
        if not broad:
            return None

        if not strict_embed:
            return broad[0]

        emb = self._videos_embeddable_batch([h.video_id for h in broad])
        for h in broad:
            if emb.get(h.video_id):
                return h
        return None

    def _is_embeddable(self, video_id: str) -> bool:
        """Validate with videos.list -> status.embeddable (uses batch cache)."""
        video_id = (video_id or "").strip()
        if not video_id or not _YT_ID_RE.match(video_id):
            return False
        return bool(self._videos_embeddable_batch([video_id]).get(video_id, False))


@lru_cache(maxsize=1)
def default_resolver() -> YouTubeTrailerResolver:
    return YouTubeTrailerResolver()
