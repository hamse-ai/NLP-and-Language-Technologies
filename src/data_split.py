"""Builds the shared train/validation split used by all five approaches. Run: python src/data_split.py"""
from pathlib import Path

from sklearn.model_selection import train_test_split

from preprocessing import LABEL_TO_IDX, load_and_clean

ROOT = Path(__file__).resolve().parent.parent
RAW_TRAIN = ROOT / "data" / "raw" / "Train.csv"
OUT_DIR = ROOT / "data" / "processed"


def main(seed: int = 42, val_size: float = 0.15):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_and_clean(RAW_TRAIN, is_train=True)
    df["label_idx"] = df["label"].map(LABEL_TO_IDX)
    if df["label_idx"].isna().any():
        unexpected = sorted(df.loc[df["label_idx"].isna(), "label"].unique().tolist())
        raise ValueError(f"Training data contains labels outside {-1, 0, 1}: {unexpected}")

    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=seed,
        stratify=df["label_idx"],
    )
    overlap = set(train_df["clean_text"]).intersection(val_df["clean_text"])
    if overlap:
        raise RuntimeError(
            f"Train/validation leakage: {len(overlap)} cleaned texts occur in both splits"
        )

    train_df.to_csv(OUT_DIR / "train_split.csv", index=False)
    val_df.to_csv(OUT_DIR / "val_split.csv", index=False)

    print(f"Total usable rows after cleaning/dedup: {len(df)}")
    print(f"Train: {len(train_df)}  Val: {len(val_df)}")
    print("Train class balance:\n", train_df["label"].value_counts(normalize=True))
    print("Val class balance:\n", val_df["label"].value_counts(normalize=True))


if __name__ == "__main__":
    main()
