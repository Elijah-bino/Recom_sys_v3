from __future__ import annotations

import datetime
import os

import numpy as np
import pandas as pd


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name, default) or default).strip()


def classic_intent(query: str | None) -> bool:
    if not query or not str(query).strip():
        return False
    q = str(query).lower()
    needles = (
        "classic",
        "classics",
        "old movie",
        "old film",
        "older film",
        "retro",
        "vintage",
        "golden age",
        "black and white",
        "silent era",
        "film noir",
        "noir",
        "70s",
        " 60s",
        " 50s",
        " 40s",
        "80s",
        "90s",
        "before 2000",
        "pre-2000",
    )
    return any(n in q for n in needles)


def recency_pool_k(top_k: int, universe: int) -> int:
    mult = int(_env("RECENCY_POOL_MULT", "15"))
    mult = max(5, min(mult, 40))
    cap = int(_env("RECENCY_POOL_CAP", "400"))
    cap = max(50, min(cap, universe))
    return min(cap, max(top_k * mult, top_k + 50))


def _years_since_release(df: pd.DataFrame, idx: int, current_year: int) -> float:
    if "release_year" not in df.columns:
        return 35.0
    y = df.loc[idx, "release_year"]
    try:
        if y is None or (isinstance(y, float) and pd.isna(y)):
            return 35.0
        yi = int(float(y))
    except (TypeError, ValueError):
        return 35.0
    if yi < 1880 or yi > current_year + 2:
        return 35.0
    return float(max(0, current_year - yi))


def apply_recency_rerank(
    df: pd.DataFrame,
    picked: np.ndarray,
    scores: np.ndarray,
    *,
    top_k: int,
    query: str | None,
    year_filter_active: bool,
    prefer_recent: bool,
) -> tuple[np.ndarray, np.ndarray]:
    alpha = float(_env("RECENCY_ALPHA", "0.42"))
    half_life = float(_env("RECENCY_HALF_LIFE_YEARS", "10"))
    if not prefer_recent or alpha <= 0.0:
        return picked[:top_k], scores[:top_k]
    if year_filter_active:
        return picked[:top_k], scores[:top_k]
    if query and classic_intent(query):
        return picked[:top_k], scores[:top_k]

    cy = datetime.date.today().year
    idxs = picked.astype(np.int64)
    sim = scores.astype(np.float32)
    years_since = np.array([_years_since_release(df, int(i), cy) for i in idxs.tolist()], dtype=np.float32)
    fresh = np.exp(-years_since / max(half_life, 1e-3))
    adjusted = sim * (1.0 + alpha * fresh)
    order = np.argsort(-adjusted)
    n = min(top_k, len(order))
    keep = order[:n]
    return idxs[keep], adjusted[keep].astype(np.float32)
