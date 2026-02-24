import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from config import (
    MODELS_DIR,
    BALANCED_DATASET,
    TARGET_COL,
    LEAKAGE_COLS,
    ID_COLS,
    OPTIONAL_DROP_COLS,
)

RANDOM_STATE = 42


@dataclass
class TrainResult:
    name: str
    model: Pipeline
    accuracy: float
    f1_macro: float


def load_dataset() -> pd.DataFrame:
    if not BALANCED_DATASET.exists():
        raise FileNotFoundError(
            f"Balanced dataset not found at {BALANCED_DATASET}\n"
            "Make sure the balanced CSV is inside the data/ folder."
        )

    print(f"[INFO] Using BALANCED dataset: {BALANCED_DATASET}")
    return pd.read_csv(BALANCED_DATASET)


def prepare_data(df: pd.DataFrame):
    drop_cols = [c for c in (LEAKAGE_COLS + ID_COLS + OPTIONAL_DROP_COLS) if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    df = df.dropna(subset=[TARGET_COL])

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(str)

    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])


def evaluate_model(name: str, pipe: Pipeline, X_test, y_test) -> TrainResult:
    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)

    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    f1_macro = report["macro avg"]["f1-score"]

    print("\n" + "=" * 70)
    print(f"MODEL: {name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 (macro): {f1_macro:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

    return TrainResult(name=name, model=pipe, accuracy=acc, f1_macro=f1_macro)


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    preprocessor = build_preprocessor(X_train)

    # ---------------------------
    # Logistic Regression (Baseline)
    # ---------------------------
    lr = LogisticRegression(max_iter=2000)
    lr_pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", lr)
    ])

    lr_pipe.fit(X_train, y_train)
    r1 = evaluate_model("Logistic Regression", lr_pipe, X_test, y_test)

    # ---------------------------
    # Random Forest + Tuning
    # ---------------------------
    print("\n[INFO] Tuning Random Forest...")

    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)

    rf_pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", rf)
    ])

    param_dist = {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [None, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2"],
    }

    search = RandomizedSearchCV(
        rf_pipe,
        param_distributions=param_dist,
        n_iter=10,
        scoring="f1_macro",
        cv=3,
        verbose=1,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    print("\nBest Parameters:")
    print(search.best_params_)

    best_rf = search.best_estimator_
    r2 = evaluate_model("Random Forest (Tuned)", best_rf, X_test, y_test)

    # Select Best Model Based on F1 Macro
    best = max([r1, r2], key=lambda r: r.f1_macro)

    model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best.model, model_path)

    print("\n" + "=" * 70)
    print(f"[DONE] Best Model: {best.name}")
    print(f"[SAVED] Model saved to: {model_path}")


if __name__ == "__main__":
    main()