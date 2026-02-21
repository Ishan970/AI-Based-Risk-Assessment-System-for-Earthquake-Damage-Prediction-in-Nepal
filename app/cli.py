import joblib
import pandas as pd
from pathlib import Path
from src.utils_explain import get_top_features, humanize_feature_name

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"

# Allowed categories (taken from your dataset; add more later if needed)
LAND_SURFACE = ["Flat", "Moderate slope", "Steep slope"]
POSITION = ["Not attached", "Attached-1 side", "Attached-2 side", "Attached-3 side"]
PLAN_CONFIG = ["Rectangular", "Square", "L-shape", "T-shape", "U-shape", "H-shape", "E-shape", "Others"]

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
    # Show numbered options
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
    # Grade 1 low, Grade 5 highest
    mapping = {
        "Grade 1": "Low Risk",
        "Grade 2": "Moderate Risk",
        "Grade 3": "High Risk",
        "Grade 4": "Very High Risk",
        "Grade 5": "Extreme Risk",
    }
    return mapping.get(damage_grade, "Unknown Risk")

def main():
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("Run training first: python src/train.py")
        return

    model = joblib.load(MODEL_PATH)
    print("\nAI-Based Risk Assessment System (Nepal Earthquake Damage Prediction)")
    print("Enter building details (pre-earthquake features).\n")

    # --- Inputs (pre-earthquake only) ---
    count_floors_pre_eq = ask_int("Number of floors (before earthquake): ", min_val=1, max_val=10)
    age_building = ask_int("Building age (years): ", min_val=0, max_val=200)
    plinth_area_sq_ft = ask_float("Plinth area (sq ft): ", min_val=50, max_val=10000)
    height_ft_pre_eq = ask_float("Building height (ft, before earthquake): ", min_val=5, max_val=200)

    land_surface_condition = ask_choice("Land surface condition:", LAND_SURFACE)

    # For these, we’ll allow free text to avoid category mismatch issues.
    # (Your model pipeline uses OneHotEncoder(handle_unknown='ignore'), so unseen values won’t crash.)
    foundation_type = input("Foundation type (e.g., Mud mortar-Stone/Brick, Cement mortar-Stone/Brick, RC): ").strip()
    roof_type = input("Roof type (e.g., Bamboo/Timber-Light roof, RCC/RB/RBC): ").strip()
    ground_floor_type = input("Ground floor type (e.g., Mud, RC, Brick/Stone): ").strip()
    other_floor_type = input("Other floor type (e.g., Timber-Planck, TImber/Bamboo-Mud): ").strip()

    position = ask_choice("Building position:", POSITION)
    plan_configuration = input("Plan configuration (e.g., Rectangular, Square, L-shape): ").strip()

    # --- Build dataframe in same column names as training ---
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

    # Predict + confidence
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

    top_feats = get_top_features(5)
    if top_feats:
        print("Top influential features in the model (global explanation):")
        for f in top_feats:
            print(f" - {humanize_feature_name(f)}")



if __name__ == "__main__":
    main()