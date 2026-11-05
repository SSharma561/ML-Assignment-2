"""
Dataset : Breast Cancer Wisconsin (Diagnostic) - UCI Machine Learning Repository
          https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
Features: 30 numeric features (>= 12 required)
Rows    : 569 instances    (>= 500 required)
Target  : diagnosis -> 1 = Malignant, 0 = Benign (binary classification)

scikit-learn ships an exact copy of this UCI dataset, so we load it from there
(no internet / Kaggle login needed) and save it as a CSV inside data/.
"""

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "breast_cancer.csv"


def main() -> None:
    raw = load_breast_cancer(as_frame=True)
    df = raw.frame.copy()

    # Clean column names: "mean radius" -> "mean_radius"
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    # sklearn uses 0 = malignant, 1 = benign. Flip it so that the
    # positive class (1) is the one we care about: Malignant.
    df["diagnosis"] = 1 - df.pop("target")

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)

    print(f"Saved {DATA_PATH.relative_to(ROOT)}")
    print(f"Shape : {df.shape}  ({df.shape[1] - 1} features + 1 target)")
    print("Class balance :")
    print(df["diagnosis"].value_counts().rename({1: "Malignant (1)", 0: "Benign (0)"}))


if __name__ == "__main__":
    main()