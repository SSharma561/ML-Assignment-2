import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from metrics import evaluate_model  # noqa: E402
from models import MODELS  # noqa: E402
from preprocessing import (  # noqa: E402
    MODEL_DIR,
    load_data,
    save_feature_names,
    save_test_data,
    split_data,
)

"""
Train all 5 models, evaluate on the held-out test set,
and save everything the Streamlit app needs.

Outputs:
    model/<model>.pkl          - 5 trained pipelines (scaler + classifier)
    model/feature_names.json   - expected input columns
    model/metrics.csv          - comparison table (used in README + app)
    test_data.csv              - test split to upload into the app
"""
def main() -> None:
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}\n")

    MODEL_DIR.mkdir(exist_ok=True)
    save_feature_names(X_train.columns)
    save_test_data(X_test, y_test)

    results = []
    for name, (factory, filename) in MODELS.items():
        model = factory()
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_DIR / filename)

        scores = evaluate_model(model, X_test, y_test)
        results.append({"ML Model Name": name, **scores})
        print(f"{name:<24} trained -> model/{filename}")

    table = pd.DataFrame(results).round(4)
    table.to_csv(MODEL_DIR / "metrics.csv", index=False)

    print("\n=== Comparison table (test set) ===")
    print(table.to_string(index=False))
    try:  # needs the optional 'tabulate' package
        print("\nMarkdown version for README:\n")
        print(table.to_markdown(index=False))
    except ImportError:
        pass


if __name__ == "__main__":
    main()