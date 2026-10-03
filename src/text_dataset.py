"""Vocabulary + PyTorch Dataset for the word-level neural models (BiLSTM, CNN-BiGRU)."""
from __future__ import annotations

from collections import Counter

import torch
from torch.utils.data import Dataset

PAD, UNK = "<pad>", "<unk>"


def simple_tokenize(text: str) -> list[str]:
    return text.lower().split()


class Vocab:
    def __init__(self, texts: list[str], min_freq: int = 2, max_size: int = 20000):
        counter = Counter()
        for t in texts:
            counter.update(simple_tokenize(t))
        self.itos = [PAD, UNK] + [
            w for w, c in counter.most_common(max_size) if c >= min_freq
        ]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, text: str, max_len: int) -> list[int]:
        ids = [self.stoi.get(tok, self.stoi[UNK]) for tok in simple_tokenize(text)][:max_len]
        ids = ids + [self.stoi[PAD]] * (max_len - len(ids))
        return ids


class TweetDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int] | None, vocab: Vocab, max_len: int = 32):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = torch.tensor(self.vocab.encode(self.texts[idx], self.max_len), dtype=torch.long)
        if self.labels is not None:
            return ids, torch.tensor(self.labels[idx], dtype=torch.long)
        return ids
