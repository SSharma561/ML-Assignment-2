"""
Step 2 - Quick Exploratory Data Analysis.
Prints basic stats and saves two plots into reports/.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no GUI needed (works on Virtual Lab / servers)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "breast_cancer.csv"
REPORT_DIR = ROOT / "reports"


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    REPORT_DIR.mkdir(exist_ok=True)

    print("Shape:", df.shape)
    print("\nMissing values in total:", int(df.isna().sum().sum()))
    print("\nTarget distribution (%):")
    print((df["diagnosis"].value_counts(normalize=True) * 100).round(2))
    print("\nSummary statistics (first 10 features):")
    print(df.iloc[:, :10].describe().T.round(3))

    # 1. Class balance
    plt.figure(figsize=(5, 4))
    sns.countplot(x="diagnosis", data=df, hue="diagnosis", palette="Set2", legend=False)
    plt.xticks([0, 1], ["Benign (0)", "Malignant (1)"])
    plt.title("Class distribution")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "class_distribution.png", dpi=120)
    plt.close()

    # 2. Correlation heatmap of the 10 "mean_*" features + target
    cols = [c for c in df.columns if c.startswith("mean_")] + ["diagnosis"]
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[cols].corr(), cmap="coolwarm", annot=True, fmt=".2f", square=True)
    plt.title("Correlation of mean_* features with diagnosis")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "correlation_heatmap.png", dpi=120)
    plt.close()

    print(f"\nPlots saved to {REPORT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()