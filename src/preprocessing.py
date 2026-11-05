import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "breast_cancer.csv"
MODEL_DIR = ROOT / "model"
TEST_DATA_PATH = ROOT / "test_data.csv"

TARGET = "diagnosis"
TEST_SIZE = 0.20
RANDOM_STATE = 42

"""
Preprocessing helpers: load data, split into train/test,
and export the test split as test_data.csv (used by the Streamlit app).
"""
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop_duplicates().dropna()
    return df


def split_data(df: pd.DataFrame):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    # stratify keeps the same Malignant/Benign ratio in train and test
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


def save_test_data(X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """test_data.csv = all input features + target column."""
    test_df = X_test.copy()
    test_df[TARGET] = y_test
    test_df.to_csv(TEST_DATA_PATH, index=False)


def save_feature_names(columns) -> None:
    """The app uses this file to validate uploaded CSVs."""
    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_DIR / "feature_names.json", "w") as f:
        json.dump({"features": list(columns), "target": TARGET}, f, indent=2)


if __name__ == "__main__":
    data = load_data()
    X_train, X_test, y_train, y_test = split_data(data)
    print("Train:", X_train.shape, " Test:", X_test.shape)