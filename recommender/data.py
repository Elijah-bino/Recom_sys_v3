from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import re


@dataclass(frozen=True)
class Catalog:
    df: pd.DataFrame
    genre_cols: list[str]


def _split_genres(s: str) -> list[str]:
    if not isinstance(s, str) or not s.strip():
        return []
    return [g.strip() for g in s.split(",") if g.strip()]


_YEAR_RE = re.compile(r"(\d{4})")


def _coerce_release_year(x) -> int | None:
    """
    Coerce values like:
    - 2019
    - "2011–2019" (en dash)
    - "2011-2019"
    - "2011–" / "2011–present"
    into a single integer year (the starting year).
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    if not s:
        return None
    m = _YEAR_RE.search(s)
    if not m:
        return None
    try:
        y = int(m.group(1))
    except Exception:
        return None
    return y if 1800 <= y <= 2100 else None


def load_catalog(catalog_csv: str | Path, limit_rows: int | None = None) -> Catalog:
    """
    Loads `catalog_meta.csv` and prepares:
    - normalized `imdb_id` column
    - `text` column built from Synopsis (+ Title as extra signal)
    - multi-hot genre columns
    """
    catalog_csv = Path(catalog_csv)
    # utf-8-sig: UTF-8 + strip BOM (common when saving from Excel on Windows)
    df = pd.read_csv(catalog_csv, encoding="utf-8-sig")
    if limit_rows is not None:
        df = df.head(int(limit_rows)).copy()

    # Normalize important columns
    df = df.rename(
        columns={
            "IMDb ID": "imdb_id",
            "Release Year": "release_year",
            "Synopsis": "synopsis",
            "Genre": "genre",
            "Title": "title",
        }
    )

    for col in ["imdb_id", "title", "synopsis", "genre"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column `{col}` in {catalog_csv}")

    df["imdb_id"] = df["imdb_id"].astype(str).str.strip()
    df["title"] = df["title"].astype(str).str.strip()
    df["synopsis"] = df["synopsis"].fillna("").astype(str)
    df["genre"] = df["genre"].fillna("").astype(str)

    # Normalize release_year into an int when present (defensive against "2011–2019").
    if "release_year" in df.columns:
        df["release_year"] = df["release_year"].apply(_coerce_release_year)

    # Text used by the deep encoder (synopsis is primary, title helps a bit)
    df["text"] = (df["title"] + ". " + df["synopsis"]).str.strip()

    # Genre multi-hot
    all_genres = sorted({g for s in df["genre"].tolist() for g in _split_genres(s)})
    genre_cols = [f"genre__{g}" for g in all_genres]
    if all_genres:
        gmat = np.zeros((len(df), len(all_genres)), dtype=np.float32)
        genre_to_idx = {g: i for i, g in enumerate(all_genres)}
        for r, s in enumerate(df["genre"].tolist()):
            for g in _split_genres(s):
                j = genre_to_idx.get(g)
                if j is not None:
                    gmat[r, j] = 1.0
        for j, g in enumerate(all_genres):
            df[f"genre__{g}"] = gmat[:, j]
    else:
        gmat = np.zeros((len(df), 0), dtype=np.float32)

    return Catalog(df=df, genre_cols=genre_cols)


def resolve_seed_items(
    catalog: Catalog,
    liked_imdb_ids: Iterable[str] | None = None,
    liked_titles: Iterable[str] | None = None,
) -> np.ndarray:
    """
    Returns row indices in catalog for provided IMDb IDs and/or titles.
    Title matching is case-insensitive exact match (fast and predictable).
    """
    idxs: list[int] = []
    df = catalog.df

    if liked_imdb_ids:
        wanted = {str(x).strip() for x in liked_imdb_ids if str(x).strip()}
        if wanted:
            hit = df.index[df["imdb_id"].isin(wanted)].tolist()
            idxs.extend(hit)

    if liked_titles:
        title_map = (
            df.reset_index()[["index", "title"]]
            .assign(_t=lambda x: x["title"].str.lower())
            .set_index("_t")["index"]
        )
        for t in liked_titles:
            key = str(t).strip().lower()
            if not key:
                continue
            if key in title_map.index:
                idxs.append(int(title_map.loc[key]))

    if not idxs:
        return np.array([], dtype=np.int64)

    return np.unique(np.array(idxs, dtype=np.int64))

