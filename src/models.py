from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

"""
Model definitions.

Every model is wrapped in a Pipeline with StandardScaler, so the scaler is
saved INSIDE each .pkl file. The Streamlit app can feed raw CSV values
directly to the loaded model - no separate preprocessing step needed.
"""
def logistic_regression():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])


# Display name -> (factory function, file name inside model/)
MODELS = {
    "Logistic Regression": (logistic_regression, "logistic_regression.pkl"),
}