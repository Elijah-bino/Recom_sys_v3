from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from recommender.data import load_catalog, resolve_seed_items
from recommender.model import Recommender
from recommender.recency_rank import apply_recency_rerank, recency_pool_k


def _print_results(df: pd.DataFrame, idx, scores):
    out = df.loc[idx, ["title", "imdb_id", "genre", "synopsis"]].copy()
    out.insert(0, "score", scores)
    # Keep synopsis short for console
    out["synopsis"] = out["synopsis"].astype(str).str.slice(0, 220)
    with pd.option_context("display.max_colwidth", 260, "display.width", 140):
        print(out.to_string(index=False))


def main():
    p = argparse.ArgumentParser(description="Deep content-based recommender (Synopsis + Genre).")
    p.add_argument("--catalog", default="catalog_meta.csv", help="Path to catalog_meta.csv")
    p.add_argument("--cache", default=".cache/item_matrix.npy", help="Where to cache item embedding matrix")
    p.add_argument("--topk", type=int, default=20, help="Number of recommendations to return")
    p.add_argument("--limit", type=int, default=None, help="Optional: only load first N rows (for quick tests)")
    p.add_argument(
        "--prefer-recent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Boost newer titles after similarity (default: on). Use --no-prefer-recent for classic-era results.",
    )

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--liked-imdb", nargs="+", help="One or more IMDb IDs the user liked (e.g. tt0111161)")
    g.add_argument("--liked-title", nargs="+", help="One or more exact titles the user liked")
    g.add_argument("--query", type=str, help="Free-text description of what the user wants")

    args = p.parse_args()

    catalog = load_catalog(args.catalog, limit_rows=args.limit)
    rec = Recommender(catalog=catalog)
    cache_path = args.cache
    if args.limit is not None:
        # Avoid mixing cache from different sizes
        cache_path = str(Path(args.cache).with_name(f"item_matrix_limit{args.limit}.npy"))
    item_matrix = rec.build_item_matrix(cache_path=cache_path)

    pool_k = recency_pool_k(args.topk, len(catalog.df))
    year_active = False  # CLI has no filter UI; set env RECENCY_ALPHA=0 to disable

    if args.query:
        idx, scores = rec.recommend_from_query(args.query, top_k=pool_k, item_matrix=item_matrix)
        idx, scores = apply_recency_rerank(
            catalog.df,
            idx,
            scores,
            top_k=args.topk,
            query=args.query,
            year_filter_active=year_active,
            prefer_recent=args.prefer_recent,
        )
        _print_results(catalog.df, idx, scores)
        return

    liked_idx = resolve_seed_items(catalog, liked_imdb_ids=args.liked_imdb, liked_titles=args.liked_title)
    if liked_idx.size == 0:
        raise SystemExit("No liked items matched. Check IMDb IDs or use exact Title matches, or use --query.")

    idx, scores = rec.recommend_from_liked_items(liked_idx, top_k=pool_k, item_matrix=item_matrix)
    idx, scores = apply_recency_rerank(
        catalog.df,
        idx,
        scores,
        top_k=args.topk,
        query=None,
        year_filter_active=year_active,
        prefer_recent=args.prefer_recent,
    )
    _print_results(catalog.df, idx, scores)


if __name__ == "__main__":
    main()

