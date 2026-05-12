from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from tqdm import tqdm

from .data import Catalog


@dataclass
class Recommender:
    """
    Two-part item representation:
    - deep text embedding of (Title + Synopsis) from a pretrained transformer
    - genre multi-hot vector

    We combine them by concatenation + L2-normalize for cosine ranking.
    This is deep-learning based and works immediately without needing user-item logs.
    """

    catalog: Catalog
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    text_model_path: Optional[str | Path] = None
    device: str | None = None

    _text_model: SentenceTransformer | None = None
    _item_matrix: np.ndarray | None = None  # shape: [N, D]

    def _get_device(self) -> str:
        if self.device:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _get_text_model(self) -> SentenceTransformer:
        if self._text_model is None:
            if self.text_model_path is None:
                self._text_model = SentenceTransformer(self.text_model_name, device=self._get_device())
            else:
                self._text_model = SentenceTransformer(str(self.text_model_path), device=self._get_device())
        return self._text_model

    def build_item_matrix(
        self,
        batch_size: int = 256,
        cache_path: str | Path | None = None,
        force_rebuild: bool = False,
    ) -> np.ndarray:
        """
        Builds and caches the item embedding matrix (for fast recommendations).
        """
        if self._item_matrix is not None and not force_rebuild:
            return self._item_matrix

        if cache_path is not None:
            cache_path = Path(cache_path)
            if cache_path.exists() and not force_rebuild:
                mat = np.load(cache_path)
                self._item_matrix = mat
                return mat

        df = self.catalog.df
        texts = df["text"].fillna("").astype(str).tolist()
        text_model = self._get_text_model()

        text_embs: list[np.ndarray] = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding synopsis+title"):
            batch = texts[i : i + batch_size]
            emb = text_model.encode(batch, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=False)
            text_embs.append(emb.astype(np.float32))
        text_emb = np.vstack(text_embs)

        # Genre features (already multi-hot float columns)
        if self.catalog.genre_cols:
            genre_mat = df[self.catalog.genre_cols].to_numpy(dtype=np.float32)
        else:
            genre_mat = np.zeros((len(df), 0), dtype=np.float32)

        mat = np.hstack([text_emb, genre_mat]).astype(np.float32)
        mat = normalize(mat, norm="l2", axis=1, copy=False)

        self._item_matrix = mat
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, mat)
        return mat

    def recommend_from_query(
        self,
        query_text: str,
        top_k: int = 20,
        item_matrix: np.ndarray | None = None,
        exclude_indices: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Recommend items similar to a free-text description of what the user wants.
        Returns (indices, scores).
        """
        item_matrix = item_matrix if item_matrix is not None else self.build_item_matrix()
        exclude = set(map(int, exclude_indices.tolist())) if exclude_indices is not None and len(exclude_indices) else set()

        text_model = self._get_text_model()
        q = text_model.encode([query_text], convert_to_numpy=True, normalize_embeddings=False).astype(np.float32)[0]

        if self.catalog.genre_cols:
            qg = np.zeros((len(self.catalog.genre_cols),), dtype=np.float32)
            # (optional) very light heuristic: if genre token appears in query, turn on that genre
            ql = (query_text or "").lower()
            for j, col in enumerate(self.catalog.genre_cols):
                g = col.replace("genre__", "").lower()
                if g and g in ql:
                    qg[j] = 1.0
            q = np.concatenate([q, qg], axis=0)

        q = normalize(q.reshape(1, -1), norm="l2", axis=1)[0]
        scores = item_matrix @ q

        if exclude:
            scores[list(exclude)] = -1e9

        k = min(top_k, len(scores))
        idx = np.argpartition(-scores, kth=k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return idx.astype(np.int64), scores[idx].astype(np.float32)

    def recommend_from_liked_items(
        self,
        liked_indices: np.ndarray,
        top_k: int = 20,
        item_matrix: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        User embedding = mean of liked item vectors.
        Returns (indices, scores) excluding already-liked items.
        """
        item_matrix = item_matrix if item_matrix is not None else self.build_item_matrix()
        liked_indices = liked_indices.astype(np.int64)
        if liked_indices.size == 0:
            raise ValueError("liked_indices is empty. Provide at least 1 liked title/IMDb ID, or use query mode.")

        u = item_matrix[liked_indices].mean(axis=0, dtype=np.float32)
        u = normalize(u.reshape(1, -1), norm="l2", axis=1)[0]
        scores = item_matrix @ u
        scores[liked_indices] = -1e9

        k = min(top_k, len(scores))
        idx = np.argpartition(-scores, kth=k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return idx.astype(np.int64), scores[idx].astype(np.float32)

