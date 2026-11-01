"""Train, evaluate and save five classifiers on the Dry Bean dataset."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from evaluation import METRIC_NAMES, compute_metrics

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "Dry_Bean_Dataset.csv"
MODEL_DIR = PROJECT_DIR / "model"
TEST_DATA_FILE = PROJECT_DIR / "test_data.csv"

TARGET_COLUMN = "Class"

RANDOM_SEED = 42
TEST_FRACTION = 0.20


def load_dataset():
    """Load the dataset and remove exact duplicate rows."""
    beans = pd.read_csv(DATA_FILE)
    n_before = len(beans)

    # Remove duplicates before splitting to avoid identical train/test examples.
    beans = beans.drop_duplicates().reset_index(drop=True)
    print(f"Loaded {n_before} rows, {n_before - len(beans)} duplicates dropped "
          f"-> {len(beans)} rows, {beans.shape[1] - 1} features")
    return beans


def build_model_grid():
    """Define the model pipelines and cross-validation parameter grids."""
    return {
        "Logistic Regression": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=5000)),
            ]),

            {"clf__C": [0.1, 1, 10, 100, 1000]},
        ),
        "Decision Tree": (
            Pipeline([
                ("clf", DecisionTreeClassifier(random_state=RANDOM_SEED)),
            ]),

            {
                "clf__criterion": ["gini", "entropy"],
                "clf__max_depth": [6, 8, 10, 12, None],
                "clf__min_samples_leaf": [1, 5, 10, 20],
            },
        ),
        "KNN": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier()),
            ]),

            {
                "clf__n_neighbors": [3, 5, 9, 15, 21, 31, 41, 61],
                "clf__weights": ["uniform", "distance"],
            },
        ),
        "Naive Bayes": (
            Pipeline([
                ("clf", GaussianNB()),
            ]),

            {"clf__var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5, 1e-3]},
        ),
        "SVM": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf")),
            ]),

            {
                "clf__C": [1, 10, 100, 1000],
                "clf__gamma": ["scale", 0.01, 0.1],
            },
        ),
    }


def file_name_for(model_name):
    """Convert a model name to its saved artifact filename."""
    return model_name.lower().replace(" ", "_") + ".joblib"


def main():
    beans = load_dataset()
    X = beans.drop(columns=[TARGET_COLUMN])
    y = beans[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_FRACTION, stratify=y, random_state=RANDOM_SEED
    )
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}\n")

    cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    MODEL_DIR.mkdir(exist_ok=True)
    results = []

    for model_name, (pipeline, param_grid) in build_model_grid().items():
        print(f"=== {model_name} ===")

        # Fit preprocessing and tune parameters within training folds only.
        search = GridSearchCV(
            pipeline, param_grid, scoring="f1_macro", cv=cv_folds, n_jobs=-1
        )
        search.fit(X_train, y_train)
        print(f"Best params : {search.best_params_}")
        print(f"CV macro-F1 : {search.best_score_:.4f}")

        best_model = search.best_estimator_

        # Calibrate the selected SVM to obtain class probabilities for AUC.
        if model_name == "SVM":
            tuned_svc = clone(best_model.named_steps["clf"])
            best_model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", CalibratedClassifierCV(tuned_svc, method="sigmoid",
                                               cv=cv_folds, ensemble=False)),
            ])
            best_model.fit(X_train, y_train)

        scores = compute_metrics(best_model, X_test, y_test)
        print("Test        : " + ", ".join(f"{k} {v:.4f}" for k, v in scores.items()) + "\n")

        joblib.dump(best_model, MODEL_DIR / file_name_for(model_name))
        results.append({"Model": model_name, **scores})

    metrics_table = pd.DataFrame(results)[["Model"] + METRIC_NAMES]
    metrics_table.to_csv(MODEL_DIR / "metrics.csv", index=False)

    test_data = X_test.copy()
    test_data[TARGET_COLUMN] = y_test
    test_data.to_csv(TEST_DATA_FILE, index=False)

    print("Test-set comparison:")
    print(metrics_table.round(4).to_string(index=False))
    print(f"\nSaved models + metrics.csv to {MODEL_DIR}")
    print(f"Saved test set to {TEST_DATA_FILE}")


if __name__ == "__main__":
    main()
