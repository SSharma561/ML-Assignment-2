from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

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

def decision_tree():
    # Trees don't need scaling; kept for a uniform pipeline structure.
    # max_depth limits over-fitting on this small dataset.
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)),
    ])

def knn():
    # KNN is distance based -> scaling is essential
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=5)),
    ])

def naive_bayes():
    # Gaussian NB because all features are continuous
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GaussianNB()),
    ])

def svm():
    # RBF kernel. AUC is computed from decision_function (see metrics.py),
    # so probability=True (deprecated in newer scikit-learn) is not needed.
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=1.0, random_state=RANDOM_STATE)),
    ])


# Display name -> (factory function, file name inside model/)
MODELS = {
    "Logistic Regression": (logistic_regression, "logistic_regression.pkl"),
    "Decision Tree": (decision_tree, "decision_tree.pkl"),
    "KNN": (knn, "knn.pkl"),
    "Naive Bayes": (naive_bayes, "naive_bayes.pkl"),
    "Support Vector Machine": (svm, "svm.pkl"),
}