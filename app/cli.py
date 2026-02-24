import joblib
import pandas as pd
from pathlib import Path
import numpy as np
import shap

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
DATASET_PATH = PROJECT_ROOT / "data" / "NepalEarthquakeDamage2015_balanced_2500.csv"

# Helper input functions
def ask_int(prompt, min_val=None, max_val=None):
    while True:
        try:
            v = int(input(prompt).strip())
            if min_val is not None and v < min_val:
                print(f"Enter a value >= {min_val}")
                continue
            if max_val is not None and v > max_val:
                print(f"Enter a value <= {max_val}")
                continue
            return v
        except ValueError:
            print("Please enter an integer.")


def ask_float(prompt, min_val=None, max_val=None):
    while True:
        try:
            v = float(input(prompt).strip())
            if min_val is not None and v < min_val:
                print(f"Enter a value >= {min_val}")
                continue
            if max_val is not None and v > max_val:
                print(f"Enter a value <= {max_val}")
                continue
            return v
        except ValueError:
            print("Please enter a number.")


def ask_choice(prompt, choices):
    """
    Shows numbered choices and returns the selected value.
    """
    print(prompt)
    for i, c in enumerate(choices, start=1):
        print(f"  {i}. {c}")
    while True:
        sel = input("Choose option number: ").strip()
        if sel.isdigit():
            idx = int(sel)
            if 1 <= idx <= len(choices):
                return choices[idx - 1]
        print("Invalid choice. Try again.")


def risk_label(damage_grade: str) -> str:
    mapping = {
        "Grade 1": "Low Risk",
        "Grade 2": "Moderate Risk",
        "Grade 3": "High Risk",
        "Grade 4": "Very High Risk",
        "Grade 5": "Extreme Risk",
    }
    return mapping.get(damage_grade, "Unknown Risk")


def get_top_features_from_csv(n=5):
    csv_path = PROJECT_ROOT / "report" / "figures" / "top_feature_importance.csv"
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    return df["feature"].head(n).tolist()


def humanize_feature_name(raw_name: str) -> str:
    name = raw_name
    if "__" in name:
        name = name.split("__", 1)[1]  # remove num__/cat__
    if raw_name.startswith("cat__") and "_" in name:
        parts = name.split("_", 1)
        if len(parts) == 2:
            return f"{parts[0]} ({parts[1]})"
    return name


def load_options():
    """
    Load unique categorical options from the balanced dataset,
    so user always selects valid categories.
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Balanced dataset not found at {DATASET_PATH}\n"
            "Put NepalEarthquakeDamage2015_balanced_2500.csv inside the data/ folder."
        )

    df = pd.read_csv(DATASET_PATH)

    def uniq(col):
        return sorted(df[col].dropna().astype(str).unique().tolist())

    options = {
        "land_surface_condition": uniq("land_surface_condition"),
        "foundation_type": uniq("foundation_type"),
        "roof_type": uniq("roof_type"),
        "ground_floor_type": uniq("ground_floor_type"),
        "other_floor_type": uniq("other_floor_type"),
        "position": uniq("position"),
        "plan_configuration": uniq("plan_configuration"),
    }

    return options

def pretty_feature_name(raw: str) -> str:
    """
    Convert sklearn ColumnTransformer names into readable feature names.
    Examples:
      num__age_building -> age_building
      cat__foundation_type_Mud mortar-Stone/Brick -> foundation_type = Mud mortar-Stone/Brick
    """
    name = raw
    if "__" in name:
        name = name.split("__", 1)[1]  # remove num__/cat__

    if raw.startswith("cat__") and "_" in name:
        base, val = name.split("_", 1)
        return f"{base} = {val}"

    return name


def main():
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("Run training first: python src/train.py")
        return

    model = joblib.load(MODEL_PATH)
    options = load_options()

    # ----- SHAP setup (local explanation) -----
    preprocess = model.named_steps["preprocess"]
    clf = model.named_steps["model"]

    feature_names = preprocess.get_feature_names_out()

    # Background data for SHAP (small sample from your balanced dataset)
    df_bg = pd.read_csv(DATASET_PATH).sample(n=100, random_state=42)

    # Drop leakage + id + target (same as training)
    drop_cols = ["count_floors_post_eq", "height_ft_post_eq", "condition_post_eq", "building_id", "district_id", "damage_grade"]
    df_bg = df_bg.drop(columns=[c for c in drop_cols if c in df_bg.columns], errors="ignore")

    X_bg_proc = preprocess.transform(df_bg)

    # Convert sparse -> dense for SHAP if needed
    if hasattr(X_bg_proc, "toarray"):
        X_bg_proc = X_bg_proc.toarray()

    explainer = shap.TreeExplainer(clf, data=X_bg_proc, feature_names=feature_names)

    print("\nAI-Based Risk Assessment System (Nepal Earthquake Damage Prediction)")
    print("Enter building details (pre-earthquake features).\n")

    # Numeric inputs
    count_floors_pre_eq = ask_int("Number of floors (before earthquake): ", min_val=1, max_val=10)
    age_building = ask_int("Building age (years): ", min_val=0, max_val=200)
    plinth_area_sq_ft = ask_float("Plinth area (sq ft): ", min_val=50, max_val=10000)
    height_ft_pre_eq = ask_float("Building height (ft, before earthquake): ", min_val=5, max_val=200)

    # Categorical inputs (numbered selection)
    land_surface_condition = ask_choice("Land surface condition:", options["land_surface_condition"])
    foundation_type = ask_choice("Foundation type:", options["foundation_type"])
    roof_type = ask_choice("Roof type:", options["roof_type"])
    ground_floor_type = ask_choice("Ground floor type:", options["ground_floor_type"])
    other_floor_type = ask_choice("Other floor type:", options["other_floor_type"])
    position = ask_choice("Building position:", options["position"])
    plan_configuration = ask_choice("Plan configuration:", options["plan_configuration"])

    # Create input row with correct column names
    row = {
        "count_floors_pre_eq": count_floors_pre_eq,
        "age_building": age_building,
        "plinth_area_sq_ft": plinth_area_sq_ft,
        "height_ft_pre_eq": height_ft_pre_eq,
        "land_surface_condition": land_surface_condition,
        "foundation_type": foundation_type,
        "roof_type": roof_type,
        "ground_floor_type": ground_floor_type,
        "other_floor_type": other_floor_type,
        "position": position,
        "plan_configuration": plan_configuration,
    }

    X_input = pd.DataFrame([row])

    pred = model.predict(X_input)[0]

    conf = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_input)[0]
        conf = probs.max()

    print("\n" + "=" * 60)
    print(f"Predicted Damage Grade: {pred}")
    print(f"Risk Level: {risk_label(pred)}")
    if conf is not None:
        print(f"Confidence: {conf*100:.2f}%")
    print("=" * 60)

    print("\nNote: Prediction is based on patterns learned from historical earthquake damage data.\n")

    # # Global explanation
    # top_feats = get_top_features_from_csv(5)
    # if top_feats:
    #     print("Top influential features in the model (global explanation):")
    #     for f in top_feats:
    #         print(f" - {humanize_feature_name(f)}")
    
    # ----- Local explanation (SHAP) -----
    X_proc = preprocess.transform(X_input)
    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()

    # For multiclass, shap_values is a list: one array per class
    shap_values = explainer.shap_values(X_proc)

    # Find which class index corresponds to predicted label
    class_list = list(clf.classes_)
    pred_class_idx = class_list.index(pred)

    sv = shap_values[pred_class_idx][0]  # shap values for this sample, predicted class
    abs_idx = np.argsort(np.abs(sv))[::-1][:5]  # top 5 features by absolute contribution

    print("\nLocal explanation (SHAP) — top factors for THIS prediction:")
    for i in abs_idx:
        fname = pretty_feature_name(feature_names[i])
        val = sv[i]
        sign = "+" if val >= 0 else "-"
        print(f" - {fname} : {sign}{abs(val):.4f}")


if __name__ == "__main__":
    main()