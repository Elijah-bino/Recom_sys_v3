from __future__ import annotations

"""
Update/refresh `catalog_meta.csv` using TMDB as the upstream source.

Design goals:
- polite: stay well below TMDB's soft limits (~40 req/s) and respect 429 Retry-After
- incremental: merge into an existing catalog (dedupe by TMDB id)
- minimal dependencies: uses requests + pandas (already in requirements.txt)

Auth:
- Prefer `TMDB_ACCESS_TOKEN` (v4 bearer token) if set
- Else use `TMDB_API_KEY` (v3 key) via query param

Sources:
- Trending:   GET /3/trending/{media_type}/{time_window}
- Popular:    GET /3/movie/popular and /3/tv/popular
- NowPlaying: GET /3/movie/now_playing
- Discover:   GET /3/discover/movie and /3/discover/tv (date window)

Docs:
- TMDB developer docs: http://developer.themoviedb.org/docs/
"""

import argparse
import os
import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Iterable, Literal

import pandas as pd
import requests

MediaType = Literal["movie", "tv", "all"]


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name, default) or default).strip()


@dataclass
class RateLimiter:
    """
    Simple token-bucket-ish limiter: ensures at most `rps` calls per second.
    """

    rps: float = 10.0  # keep comfortably below ~40 rps
    _min_interval_s: float = 0.0
    _next_ok_s: float = 0.0

    def __post_init__(self) -> None:
        self._min_interval_s = 1.0 / max(0.1, float(self.rps))
        self._next_ok_s = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        if now < self._next_ok_s:
            time.sleep(self._next_ok_s - now)
        self._next_ok_s = time.monotonic() + self._min_interval_s


class TMDBClient:
    def __init__(self, *, timeout_s: float = 12.0, rps: float = 10.0):
        self.api_key = _env("TMDB_API_KEY")
        self.access_token = _env("TMDB_ACCESS_TOKEN")
        if not (self.api_key or self.access_token):
            raise SystemExit("Set TMDB_API_KEY or TMDB_ACCESS_TOKEN in your environment.")
        self.timeout_s = float(timeout_s)
        self.rl = RateLimiter(rps=float(rps))

    def _headers(self) -> dict[str, str]:
        h = {"accept": "application/json"}
        if self.access_token:
            h["Authorization"] = f"Bearer {self.access_token}"
        return h

    def get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        """
        GET with:
        - steady-state RPS limit
        - exponential backoff on transient failures
        - strict respect for 429 Retry-After when present
        """
        url = f"https://api.themoviedb.org{path}"
        qp = dict(params)
        if not self.access_token:
            qp["api_key"] = self.api_key

        backoff_s = 0.75
        for attempt in range(1, 8):
            self.rl.wait()
            try:
                r = requests.get(url, params=qp, headers=self._headers(), timeout=self.timeout_s)
            except Exception:
                time.sleep(backoff_s)
                backoff_s = min(backoff_s * 1.8, 12.0)
                continue

            # Respect rate limiting
            if r.status_code == 429:
                ra = (r.headers.get("Retry-After") or "").strip()
                sleep_s = float(ra) if ra.replace(".", "", 1).isdigit() else max(2.0, backoff_s)
                time.sleep(min(sleep_s, 60.0))
                backoff_s = min(backoff_s * 1.6, 20.0)
                continue

            if 500 <= r.status_code < 600:
                time.sleep(backoff_s)
                backoff_s = min(backoff_s * 1.8, 12.0)
                continue

            if r.status_code != 200:
                raise RuntimeError(f"{r.status_code} {r.text[:300]}")

            try:
                return r.json() or {}
            except Exception as e:
                raise RuntimeError(f"Invalid JSON from {path}: {e}") from e

        raise RuntimeError(f"TMDB request failed after retries: {path}")


def _safe_year(d: str) -> int | None:
    if not isinstance(d, str):
        return None
    d = d.strip()
    if len(d) >= 4 and d[:4].isdigit():
        return int(d[:4])
    return None


def _normalize_item(it: dict[str, Any]) -> dict[str, Any] | None:
    """
    Normalize TMDB list item into a row compatible with our catalog.
    We keep a stable ID:
      imdb_id column becomes "tmdb:<media_type>:<id>" (no licensing issues)
    """
    mt = str(it.get("media_type") or "").strip().lower()
    # list endpoints for /movie/* don't include media_type; infer later
    if mt not in {"movie", "tv"}:
        mt = ""  # filled by caller if known

    tmdb_id = it.get("id")
    if tmdb_id is None:
        return None
    try:
        tmdb_id_i = int(tmdb_id)
    except Exception:
        return None

    title = str(it.get("title") or it.get("name") or "").strip()
    if not title:
        return None

    synopsis = str(it.get("overview") or "").strip()
    genres_ids = it.get("genre_ids") or []
    if not isinstance(genres_ids, list):
        genres_ids = []

    rd = str(it.get("release_date") or it.get("first_air_date") or "").strip()
    year = _safe_year(rd)

    return {
        "tmdb_id": tmdb_id_i,
        "tmdb_media_type": mt or None,
        "Title": title,
        "Synopsis": synopsis,
        "Release Year": year,
        # We'll fill Genre after we download the genre map
        "Genre": None,
        # Keep compatibility with existing code paths
        "IMDb ID": None,
        # Extra signals (useful for re-ranking / freshness)
        "tmdb_popularity": float(it.get("popularity") or 0.0),
        "tmdb_vote_average": float(it.get("vote_average") or 0.0),
        "tmdb_vote_count": int(it.get("vote_count") or 0),
        "tmdb_release_date": rd or None,
    }


def _fetch_genre_map(client: TMDBClient, *, language: str) -> dict[tuple[str, int], str]:
    """
    Returns mapping (media_type, genre_id) -> genre_name
    """
    out: dict[tuple[str, int], str] = {}
    for mt in ("movie", "tv"):
        data = client.get(f"/3/genre/{mt}/list", params={"language": language})
        for g in data.get("genres") or []:
            try:
                gid = int(g.get("id"))
                name = str(g.get("name") or "").strip()
            except Exception:
                continue
            if gid and name:
                out[(mt, gid)] = name
    return out


def _attach_genres(rows: list[dict[str, Any]], genre_map: dict[tuple[str, int], str], *, media_type: str, raw_items: list[dict[str, Any]]):
    """
    rows and raw_items are aligned lists.
    """
    for row, it in zip(rows, raw_items):
        mt = str(row.get("tmdb_media_type") or "").strip().lower() or media_type
        ids = it.get("genre_ids") or []
        names: list[str] = []
        if isinstance(ids, list):
            for gid in ids:
                try:
                    gidi = int(gid)
                except Exception:
                    continue
                nm = genre_map.get((mt, gidi))
                if nm:
                    names.append(nm)
        # TMDB uses a small, stable set; join for existing multi-hot logic
        row["Genre"] = ", ".join(sorted(dict.fromkeys(names))) if names else ""
        row["tmdb_media_type"] = mt
        row["IMDb ID"] = f"tmdb:{mt}:{int(row['tmdb_id'])}"


def _collect_paged(client: TMDBClient, path: str, params: dict[str, Any], *, pages: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for p in range(1, int(pages) + 1):
        data = client.get(path, params={**params, "page": p})
        res = data.get("results") or []
        if not isinstance(res, list):
            break
        items.extend([x for x in res if isinstance(x, dict)])
    return items


def build_catalog_rows(
    client: TMDBClient,
    *,
    language: str,
    region: str | None,
    pages: int,
    days_back: int,
) -> pd.DataFrame:
    genre_map = _fetch_genre_map(client, language=language)

    today = date.today()
    start = today - timedelta(days=int(days_back))
    start_s = start.isoformat()
    end_s = today.isoformat()

    all_rows: list[dict[str, Any]] = []

    def add_source(media_type: Literal["movie", "tv"], *, raw: list[dict[str, Any]]):
        rows: list[dict[str, Any]] = []
        for it in raw:
            row = _normalize_item(it)
            if row is None:
                continue
            # list endpoints sometimes omit media_type
            row["tmdb_media_type"] = row.get("tmdb_media_type") or media_type
            rows.append(row)
        _attach_genres(rows, genre_map, media_type=media_type, raw_items=raw[: len(rows)])
        all_rows.extend(rows)

    # Trending daily/weekly (new stuff bubbles up)
    trending = _collect_paged(
        client,
        "/3/trending/all/day",
        params={"language": language},
        pages=pages,
    )
    # Split trending into movie/tv by `media_type`
    movies_t = [x for x in trending if str(x.get("media_type") or "") == "movie"]
    tv_t = [x for x in trending if str(x.get("media_type") or "") == "tv"]
    add_source("movie", raw=movies_t)
    add_source("tv", raw=tv_t)

    # Popular (good general coverage)
    add_source(
        "movie",
        raw=_collect_paged(client, "/3/movie/popular", params={"language": language, **({"region": region} if region else {})}, pages=pages),
    )
    add_source(
        "tv",
        raw=_collect_paged(client, "/3/tv/popular", params={"language": language}, pages=pages),
    )

    # Now playing (theatre releases)
    add_source(
        "movie",
        raw=_collect_paged(
            client,
            "/3/movie/now_playing",
            params={"language": language, **({"region": region} if region else {})},
            pages=max(1, pages // 2),
        ),
    )

    # Discover within date window to catch recent releases even if not trending yet
    add_source(
        "movie",
        raw=_collect_paged(
            client,
            "/3/discover/movie",
            params={
                "language": language,
                "include_adult": "false",
                "include_video": "false",
                "sort_by": "primary_release_date.desc",
                "primary_release_date.gte": start_s,
                "primary_release_date.lte": end_s,
                **({"region": region} if region else {}),
            },
            pages=pages,
        ),
    )
    add_source(
        "tv",
        raw=_collect_paged(
            client,
            "/3/discover/tv",
            params={
                "language": language,
                "include_adult": "false",
                "sort_by": "first_air_date.desc",
                "first_air_date.gte": start_s,
                "first_air_date.lte": end_s,
            },
            pages=pages,
        ),
    )

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df

    # Dedupe by stable TMDB identity
    df = df.drop_duplicates(subset=["tmdb_media_type", "tmdb_id"], keep="first").reset_index(drop=True)

    # Keep the exact legacy columns first (so existing code keeps working)
    ordered = [
        "Title",
        "IMDb ID",
        "Release Year",
        "Synopsis",
        "Genre",
        "tmdb_id",
        "tmdb_media_type",
        "tmdb_release_date",
        "tmdb_popularity",
        "tmdb_vote_average",
        "tmdb_vote_count",
    ]
    for c in ordered:
        if c not in df.columns:
            df[c] = None
    return df[ordered]


def merge_into_existing(existing_path: str, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge by TMDB identity if possible; else by IMDb ID if present.
    New rows win on conflicts (fresh metadata).
    """
    if not os.path.exists(existing_path):
        return new_df

    old = pd.read_csv(existing_path, encoding="utf-8-sig")
    # If old file doesn't have tmdb columns, we still keep it, but dedupe on IMDb ID
    has_tmdb = ("tmdb_id" in old.columns) and ("tmdb_media_type" in old.columns)

    combined = pd.concat([old, new_df], ignore_index=True, sort=False)
    if has_tmdb:
        combined = combined.drop_duplicates(subset=["tmdb_media_type", "tmdb_id"], keep="last")
    elif "IMDb ID" in combined.columns:
        combined = combined.drop_duplicates(subset=["IMDb ID"], keep="last")
    else:
        combined = combined.drop_duplicates(subset=["Title", "Release Year"], keep="last")

    # Try to keep legacy column names consistent
    for col in ["Title", "IMDb ID", "Release Year", "Synopsis", "Genre"]:
        if col not in combined.columns:
            combined[col] = ""
    return combined.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Update catalog_meta.csv from TMDB (polite + incremental).")
    ap.add_argument("--out", default="catalog_meta.csv", help="Output CSV path (default: catalog_meta.csv)")
    ap.add_argument("--merge", action="store_true", help="Merge into existing --out (incremental) if it exists")
    ap.add_argument("--language", default="en-US", help="TMDB language (default: en-US)")
    ap.add_argument("--region", default="", help="Optional region (ISO-3166-1, e.g. US, AU) for some lists")
    ap.add_argument("--pages", type=int, default=10, help="Pages per source endpoint (default: 10)")
    ap.add_argument("--days-back", type=int, default=365, help="Discover window in days (default: 365)")
    ap.add_argument("--rps", type=float, default=10.0, help="Max requests per second (default: 10; stay < 40)")
    ap.add_argument("--timeout", type=float, default=12.0, help="Request timeout seconds")
    args = ap.parse_args()

    client = TMDBClient(timeout_s=float(args.timeout), rps=float(args.rps))
    new_df = build_catalog_rows(
        client,
        language=str(args.language),
        region=(str(args.region).strip().upper() or None),
        pages=int(args.pages),
        days_back=int(args.days_back),
    )
    if new_df.empty:
        raise SystemExit("No rows returned from TMDB. Check your API key/token and try again.")

    out_path = str(args.out)
    if bool(args.merge):
        final = merge_into_existing(out_path, new_df)
    else:
        final = new_df

    # Write atomically-ish
    tmp = out_path + ".tmp"
    final.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, out_path)
    print(f"Wrote {len(final):,} rows to {out_path}")


if __name__ == "__main__":
    main()

