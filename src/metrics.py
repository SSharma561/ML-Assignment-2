import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

"""
Evaluation metrics shared by the training script and the Streamlit app.
Accuracy, AUC, Precision, Recall, F1, MCC.
"""
def evaluate_model(model, X, y_true) -> dict:
    y_pred = model.predict(X)

    # AUC needs a score/probability for the positive class
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)[:, 1]
    else:
        y_score = model.decision_function(X)

    # AUC is undefined if the uploaded file contains only one class
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = np.nan

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }