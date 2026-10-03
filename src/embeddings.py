"""Loads pretrained GloVe vectors and builds an embedding matrix aligned to a project Vocab."""
from __future__ import annotations

import numpy as np


def load_glove(path: str) -> dict[str, np.ndarray]:
    """Read a GloVe .txt file (`word v1 v2 ... vd` per line) into a dict."""
    vectors = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            vectors[word] = np.asarray(parts[1:], dtype=np.float32)
    return vectors


def build_embedding_matrix(vocab, glove: dict[str, np.ndarray], embed_dim: int, seed: int = 42):
    """Embedding matrix for `vocab`: GloVe vector where available, random init otherwise."""
    rng = np.random.RandomState(seed)
    matrix = rng.normal(scale=0.1, size=(len(vocab), embed_dim)).astype(np.float32)
    hits = 0
    for word, idx in vocab.stoi.items():
        if word in glove:
            matrix[idx] = glove[word]
            hits += 1
    matrix[vocab.stoi["<pad>"]] = 0.0
    return matrix, hits
