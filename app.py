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

tab_single, tab_compare = st.tabs(["Selected model", "Compare all models"])

# ---------- Tab 1: selected model ----------
with tab_single:
    model = load_model(MODEL_FILES[model_name])
    scores = evaluate_model(model, X, y)

    st.subheader(f"Evaluation metrics - {model_name}")
    cols = st.columns(6)
    for col, (metric, value) in zip(cols, scores.items()):
        col.metric(metric, "N/A" if pd.isna(value) else f"{value:.4f}")

# ---------- Tab 1 (cont.): confusion matrix + classification report ----------
with tab_single:
    y_pred = model.predict(X)
    left, right = st.columns(2)

    with left:
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Classification report")
        report = classification_report(
            y, y_pred, labels=[0, 1], target_names=CLASS_NAMES,
            output_dict=True, zero_division=0,
        )
        st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

# ---------- Tab 2: all models ----------
with tab_compare:
    st.subheader("All 5 models on the uploaded data")
    rows = []
    for name, filename in MODEL_FILES.items():
        rows.append({"ML Model Name": name, **evaluate_model(load_model(filename), X, y)})
    table = pd.DataFrame(rows).set_index("ML Model Name").round(4)
    st.dataframe(
        table.style.highlight_max(axis=0, color="#c6efce").format("{:.4f}"),
        use_container_width=True,
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    table.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylim(0.7, 1.0)
    ax.set_ylabel("Score")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=6, fontsize=8, frameon=False)
    plt.xticks(rotation=0)
    st.pyplot(fig)
    plt.close(fig)

    best = table["F1"].idxmax()
    st.success(f"Best model on this data by F1 score: **{best}**")