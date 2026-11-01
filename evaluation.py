"""Shared evaluation metrics for training and the Streamlit app."""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def compute_metrics(model, X, y_true):
    """Compute the six metrics without fitting the model."""
    y_pred = model.predict(X)

    y_proba = model.predict_proba(X)

    # OvR AUC is undefined when any trained class is absent.
    try:
        if set(y_true) != set(model.classes_):
            raise ValueError("AUC requires all trained classes")
        auc = roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="macro", labels=model.classes_
        )
    except ValueError:
        auc = float("nan")

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def confusion_table(model, X, y_true):
    """Return a confusion matrix with actual classes as rows."""
    labels = list(model.classes_)
    matrix = confusion_matrix(y_true, model.predict(X), labels=labels)
    return pd.DataFrame(matrix, index=labels, columns=labels)


def per_class_report(model, X, y_true):
    """Return per-class precision, recall, F1 and support."""
    report = classification_report(
        y_true, model.predict(X), output_dict=True, zero_division=0
    )

    report.pop("accuracy", None)
    return pd.DataFrame(report).T
