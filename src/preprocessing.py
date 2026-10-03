"""Text cleaning shared across all five approaches, so model differences aren't just preprocessing differences."""
from __future__ import annotations

import html
import re
from pathlib import Path

import pandas as pd

USER_TOKEN = "<user>"
URL_TOKEN = "<url>"

_WHITESPACE_RE = re.compile(r"\s+")
_USER_RE = re.compile(r"<\s*user\s*>", flags=re.IGNORECASE)
_URL_RE = re.compile(r"<\s*url\s*>", flags=re.IGNORECASE)
_ELONGATION_RE = re.compile(r"([a-zA-Z])\1{2,}")


def clean_text(text: str) -> str:
    """Clean a single tweet."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = _USER_RE.sub(USER_TOKEN, text)
    text = _URL_RE.sub(URL_TOKEN, text)
    text = _ELONGATION_RE.sub(r"\1\1", text)  # soooo -> soo
    text = text.replace("\n", " ").replace("\r", " ")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def load_and_clean(csv_path: str | Path, is_train: bool) -> pd.DataFrame:
    """Load a Zindi CSV. For training data, also resolves the conflicting-label duplicates (EDA finding #3)."""
    df = pd.read_csv(csv_path)
    df["clean_text"] = df["safe_text"].apply(clean_text)

    if is_train:
        df = df.dropna(subset=["safe_text", "label"]).copy()
        df["label"] = df["label"].astype(int)

        # Keep the highest-agreement row per duplicate text instead of an arbitrary one.
        # Preserve source row order when agreements tie, so the retained row is
        # deterministic even for conflicting duplicates with equal agreement.
        df = df.sort_values("agreement", ascending=False, kind="stable")
        df = df.drop_duplicates(subset="clean_text", keep="first")
        df = df.reset_index(drop=True)
    else:
        df["clean_text"] = df["clean_text"].fillna("")

    return df


LABEL_TO_IDX = {-1: 0, 0: 1, 1: 2}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}
CLASS_NAMES = ["Negative", "Neutral", "Positive"]
