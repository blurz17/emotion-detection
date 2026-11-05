from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

COLUMNS = ["text", "labels", "comment_id"]


def load_data(split="train", **kwargs):
    """
    Load data from the TSV file (or CSV fallback) and return a pandas DataFrame.

    Args:
        split: "train" or "test" (also accepts type="train" for backward compatibility)
    """
    # Accept 'type' parameter if passed
    if "type" in kwargs:
        split = kwargs["type"]

    if split not in ["train", "test"]:
        raise ValueError(f"split must be 'train' or 'test', got '{split}'")

    # Try .tsv first, then .csv
    file_path = DATA_DIR / f"{split}.tsv"
    if not file_path.exists():
        file_path = DATA_DIR / f"{split}.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Data file for '{split}' not found in {DATA_DIR}")

    sep = "\t" if file_path.suffix == ".tsv" else ","
    return pd.read_csv(file_path, sep=sep, names=COLUMNS)


def load_emotions():
    """Load the emotion category labels from emotions.txt."""
    file_path = DATA_DIR / "emotions.txt"
    if not file_path.exists():
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
