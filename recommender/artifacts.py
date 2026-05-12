from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .data import Catalog, load_catalog
from .model import Recommender


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_stable_json(payload) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class BundlePaths:
    out_dir: Path
    text_model_dir: Path
    item_matrix_path: Path
    manifest_path: Path


def bundle_paths(out_dir: str | Path) -> BundlePaths:
    out_dir = Path(out_dir)
    return BundlePaths(
        out_dir=out_dir,
        text_model_dir=out_dir / "text_model",
        item_matrix_path=out_dir / "item_matrix.npy",
        manifest_path=out_dir / "manifest.json",
    )


def export_artifact_bundle(
    *,
    catalog_csv: str | Path,
    out_dir: str | Path,
    limit_rows: int | None,
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedding_batch_size: int = 256,
    force_rebuild_matrix: bool = False,
    device: str | None = None,
) -> BundlePaths:
    """
    Exports deployment artifacts:
    - text_model/: saved SentenceTransformer weights + tokenizer/config
    - item_matrix.npy: final normalized item vectors (deep text concat genre multi-hot)
    - manifest.json: metadata needed to validate/serve consistently
    """
    paths = bundle_paths(out_dir)
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    paths.text_model_dir.mkdir(parents=True, exist_ok=True)

    catalog: Catalog = load_catalog(catalog_csv, limit_rows=limit_rows)
    catalog_path = Path(catalog_csv)

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Save model weights locally for offline deployments:
    # 1) load from hub name (downloads into HF cache as needed)
    # 2) save a self-contained folder you can ship to prod
    src = SentenceTransformer(text_model_name, device=dev)
    src.save(str(paths.text_model_dir))

    rec = Recommender(
        catalog=catalog,
        text_model_name=text_model_name,
        text_model_path=paths.text_model_dir,
        device=device,
    )

    mat: np.ndarray
    if paths.item_matrix_path.exists() and not force_rebuild_matrix:
        mat = np.load(paths.item_matrix_path)
        rec._item_matrix = mat
    else:
        # Build matrix using bundled local weights + save to deterministic path.
        mat = rec.build_item_matrix(
            batch_size=embedding_batch_size,
            cache_path=paths.item_matrix_path,
            force_rebuild=True,
        )
        rec._item_matrix = mat

    manifest = {
        "catalog_path": str(catalog_path.resolve()),
        "catalog_rows": int(len(catalog.df)),
        "limit_rows": int(limit_rows) if limit_rows is not None else None,
        "source_text_model_name": text_model_name,
        "exported_text_model_dir": str(paths.text_model_dir.resolve()),
        "embedding_dim_concat": int(mat.shape[1]),
        "embedding_batch_size": int(embedding_batch_size),
        # lightweight integrity checks (whole-file hashes can be expensive for huge CSV)
        "catalog_mtime_ns": int(catalog_path.stat().st_mtime_ns),
        "catalog_size_bytes": int(catalog_path.stat().st_size),
        "genre_dim": int(len(catalog.genre_cols)),
        "item_matrix_sha256": hashlib.sha256(mat.tobytes()).hexdigest(),
    }
    write_manifest(paths.manifest_path, manifest)
    return paths
