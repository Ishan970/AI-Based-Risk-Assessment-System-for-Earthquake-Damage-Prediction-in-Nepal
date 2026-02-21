import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
FIG_DIR = PROJECT_ROOT / "report" / "figures"

def main(top_n: int = 15):
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("Run training first: python src/train.py")
        return

    model = joblib.load(MODEL_PATH)

    # model is a Pipeline: preprocess -> model
    preprocess = model.named_steps["preprocess"]
    clf = model.named_steps["model"]

    # Only tree-based models have feature_importances_
    if not hasattr(clf, "feature_importances_"):
        print("[INFO] This model does not support feature_importances_.")
        print("If best model is Logistic Regression, we can do coefficient-based explanation instead.")
        return

    # Get feature names after preprocessing (numeric + onehot categories)
    feature_names = preprocess.get_feature_names_out()
    importances = clf.feature_importances_

    # Sort top features
    idx = np.argsort(importances)[::-1][:top_n]
    top_features = feature_names[idx]
    top_importances = importances[idx]

    # Save table as CSV for report
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = FIG_DIR / "top_feature_importance.csv"
    pd.DataFrame({
        "feature": top_features,
        "importance": top_importances
    }).to_csv(out_csv, index=False)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(top_features[::-1], top_importances[::-1])
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Feature Importances (Random Forest)")
    plt.tight_layout()

    out_png = FIG_DIR / "feature_importance.png"
    plt.savefig(out_png, dpi=200)
    plt.close()

    print(f"[DONE] Saved feature importance plot: {out_png}")
    print(f"[DONE] Saved feature importance table: {out_csv}")

if __name__ == "__main__":
    main()