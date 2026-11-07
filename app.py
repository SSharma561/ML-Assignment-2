import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))
from metrics import evaluate_model  # noqa: E402

MODEL_DIR = ROOT / "model"
SAMPLE_TEST_PATH = ROOT / "test_data.csv"
CLASS_NAMES = ["Benign (0)", "Malignant (1)"]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "svm.pkl",
}

"""
Streamlit app - Breast Cancer Classification (ML Assignment 2)

Features
  a. Upload a test CSV (features + 'diagnosis' target column)
  b. Choose one of 5 pre-trained models from a dropdown
  c. Show Accuracy, AUC, Precision, Recall, F1, MCC
  d. Show confusion matrix and classification report
  + Compare all 5 models on the same uploaded data

Models are only LOADED from model/*.pkl - nothing is retrained here.
"""

st.set_page_config(page_title="ML Assignment 2 - Classifier Demo", layout="wide")


# ---------- Cached loaders ----------
@st.cache_resource
def load_model(filename: str):
    return joblib.load(MODEL_DIR / filename)


@st.cache_data
def load_schema() -> dict:
    with open(MODEL_DIR / "feature_names.json") as f:
        return json.load(f)


@st.cache_data
def load_sample_csv() -> bytes:
    return SAMPLE_TEST_PATH.read_bytes()


schema = load_schema()
FEATURES, TARGET = schema["features"], schema["target"]

# ---------- Sidebar ----------
st.sidebar.header("1. Upload test data")
uploaded = st.sidebar.file_uploader("CSV with all features + target", type=["csv"])
st.sidebar.download_button(
    "Download sample test_data.csv",
    data=load_sample_csv(),
    file_name="test_data.csv",
    mime="text/csv",
)

st.sidebar.header("2. Choose a model")
model_name = st.sidebar.selectbox("Model", list(MODEL_FILES.keys()))

# ---------- Header ----------
st.title("Breast Cancer Classification - 5 ML Models")
st.caption(
    "Dataset: Breast Cancer Wisconsin (Diagnostic), UCI · 30 features · "
    "target `diagnosis` (1 = Malignant, 0 = Benign)"
)

if uploaded is None:
    st.info(
        "Upload a test CSV from the sidebar to begin. "
        "You can download the sample **test_data.csv** there."
    )
    st.stop()

# ---------- Read + validate upload ----------
try:
    df = pd.read_csv(uploaded)
except Exception as e:  # noqa: BLE001
    st.error(f"Could not read the CSV file: {e}")
    st.stop()

missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing:
    st.error(f"Uploaded file is missing {len(missing)} required column(s): {missing}")
    st.stop()

df = df.dropna(subset=FEATURES + [TARGET])
X, y = df[FEATURES], df[TARGET].astype(int)

with st.expander(f"Preview uploaded data ({df.shape[0]} rows × {df.shape[1]} columns)"):
    st.dataframe(df.head(20), use_container_width=True)