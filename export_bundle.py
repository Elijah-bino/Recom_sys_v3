from __future__ import annotations

import argparse
from pathlib import Path

from recommender.artifacts import export_artifact_bundle


def main():
    p = argparse.ArgumentParser(description="Export deployable artifacts (model weights + item matrix + manifest).")
    p.add_argument("--catalog", default="catalog_meta.csv", help="Path to catalog_meta.csv")
    p.add_argument("--out", default="artifacts/bundle", help="Output directory to write the bundle into")
    p.add_argument("--limit", type=int, default=None, help="Optional: only embed first N rows (dev/test)")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="HF model id to export")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--force-rebuild-matrix", action="store_true")
    args = p.parse_args()

    paths = export_artifact_bundle(
        catalog_csv=Path(args.catalog),
        out_dir=Path(args.out),
        limit_rows=args.limit,
        text_model_name=args.model,
        embedding_batch_size=int(args.batch_size),
        force_rebuild_matrix=bool(args.force_rebuild_matrix),
    )

    print("Exported bundle:")
    print(f"- text_model: {paths.text_model_dir}")
    print(f"- item_matrix: {paths.item_matrix_path}")
    print(f"- manifest: {paths.manifest_path}")


if __name__ == "__main__":
    main()
