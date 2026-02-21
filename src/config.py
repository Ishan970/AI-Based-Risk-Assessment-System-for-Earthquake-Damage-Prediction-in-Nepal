from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Only dataset used in this project
BALANCED_DATASET = DATA_DIR / "NepalEarthquakeDamage2015_balanced_2500.csv"

TARGET_COL = "damage_grade"

# Columns that leak post-earthquake information
LEAKAGE_COLS = [
    "count_floors_post_eq",
    "height_ft_post_eq",
    "condition_post_eq",
]

# Identifier column (not useful for prediction)
ID_COLS = ["building_id"]

# Optional drop (leave empty for now)
OPTIONAL_DROP_COLS = []