from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMPORTANCE_CSV = PROJECT_ROOT / "report" / "figures" / "top_feature_importance.csv"

def get_top_features(n=5):
    """
    Returns the top-N most important features from the saved CSV.
    """
    if not IMPORTANCE_CSV.exists():
        return []
    df = pd.read_csv(IMPORTANCE_CSV)
    return df["feature"].head(n).tolist()

def humanize_feature_name(raw_name: str) -> str:
    """
    Converts sklearn feature names into something readable.
    Examples:
      num__age_building -> age_building
      cat__foundation_type_Mud mortar-Stone/Brick -> foundation_type (Mud mortar-Stone/Brick)
    """
    name = raw_name
    if "__" in name:
        name = name.split("__", 1)[1]
    if "_" in name and "cat" in raw_name:
        # OneHotEncoder outputs like: foundation_type_Mud mortar-...
        parts = name.split("_", 1)
        if len(parts) == 2:
            return f"{parts[0]} ({parts[1]})"
    return name