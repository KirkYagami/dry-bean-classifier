"""Streamlit interface for evaluating the saved Dry Bean classifiers."""

from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from evaluation import METRIC_NAMES, compute_metrics, confusion_table, per_class_report

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "model"
BUNDLED_TEST_FILE = PROJECT_DIR / "test_data.csv"
TARGET_COLUMN = "Class"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib",
}

METRIC_HINTS = {
    "Accuracy": "Share of beans classified correctly.",
    "AUC": "One-vs-rest ROC AUC, macro-averaged over the 7 varieties.",
    "Precision": "Macro avg: when the model says variety X, how often it is X.",
    "Recall": "Macro avg: of all true X beans, how many were found.",
    "F1": "Macro avg of the harmonic mean of precision and recall.",
    "MCC": "Matthews correlation over the full confusion matrix (-1 to 1).",
}

HEAT_LOW, HEAT_HIGH = "#cde2fb", "#0d366b"
SERIES_BLUE = "#2a78d6"


@st.cache_resource
def load_models():
    return {name: joblib.load(MODEL_DIR / file) for name, file in MODEL_FILES.items()}


@st.cache_data
def load_bundled_test_data():
    return pd.read_csv(BUNDLED_TEST_FILE)


@st.cache_data
def load_training_metrics():
    return pd.read_csv(MODEL_DIR / "metrics.csv")


def validate_upload(frame, feature_columns, known_classes):
    """Validate the CSV and return features in the training column order."""
    problems = []
    missing = [c for c in list(feature_columns) + [TARGET_COLUMN] if c not in frame.columns]
    if missing:
        problems.append("Missing column(s): " + ", ".join(missing))
        return None, None, problems

    if frame.empty:
        return None, None, ["The CSV contains no data rows."]
    features = frame[list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        return None, None, [
            "Feature values must be numeric and finite, with no missing values. "
            "Correct the CSV and upload it again."
        ]
    if frame[TARGET_COLUMN].isna().any():
        return None, None, ["The Class column contains missing labels."]
    labels = frame[TARGET_COLUMN].astype(str).str.strip().str.upper()
    unknown = sorted(set(labels) - set(known_classes))
    if unknown:
        return None, None, ["Unknown class label(s): " + ", ".join(unknown)]
    return features, labels, problems


def confusion_heatmap(matrix, as_percent):
    """Plot counts or percentages for each actual/predicted class pair."""
    long = matrix.reset_index(names="Actual").melt(
        id_vars="Actual", var_name="Predicted", value_name="Count"
    )
    row_totals = long.groupby("Actual")["Count"].transform("sum")
    long["Pct"] = (long["Count"] / row_totals * 100).fillna(0)
    value_field = "Pct" if as_percent else "Count"
    long["Label"] = long["Pct"].map("{:.1f}%".format) if as_percent else long["Count"].astype(str)

    order = list(matrix.index)
    base = alt.Chart(long).encode(
        x=alt.X("Predicted:N", sort=order, title="Predicted variety",
                axis=alt.Axis(labelAngle=-40)),
        y=alt.Y("Actual:N", sort=order, title="Actual variety"),
    )
    cells = base.mark_rect(stroke="white", strokeWidth=2).encode(
        color=alt.Color(f"{value_field}:Q", scale=alt.Scale(range=[HEAT_LOW, HEAT_HIGH]),
                        legend=None),
        tooltip=[
            alt.Tooltip("Actual:N"), alt.Tooltip("Predicted:N"),
            alt.Tooltip("Count:Q"), alt.Tooltip("Pct:Q", title="% of actual", format=".1f"),
        ],
    )

    threshold = long[value_field].max() * 0.5
    numbers = base.mark_text(fontSize=12).encode(
        text="Label:N",
        color=alt.condition(alt.datum[value_field] > threshold,
                            alt.value("white"), alt.value("#0b0b0b")),
    )
    return (cells + numbers).properties(height=380)


def metric_dot_plot(scores, metric, highlight):
    """Compare model scores and highlight the selected model."""
    data = scores[["Model", metric]].rename(columns={metric: "Score"}).dropna()
    data["Selected"] = data["Model"] == highlight
    if data.empty:
        return None
    low = max(-1.0 if metric == "MCC" else 0.0, data["Score"].min() - 0.03)
    data["Start"] = low

    x_scale = alt.Scale(domain=[low, 1.0], zero=False)
    base = alt.Chart(data).encode(
        y=alt.Y("Model:N", title=None,
                sort=alt.EncodingSortField(field="Score", order="descending")),
        tooltip=[alt.Tooltip("Model:N"), alt.Tooltip("Score:Q", format=".4f")],
    )
    rule = base.mark_rule(color="#a8a79f", strokeWidth=2).encode(
        x=alt.X("Start:Q", scale=x_scale, title=metric), x2="Score:Q"
    )
    dots = base.mark_point(filled=True, size=160, color=SERIES_BLUE).encode(
        x=alt.X("Score:Q", scale=x_scale),
        opacity=alt.condition(alt.datum.Selected, alt.value(1.0), alt.value(0.45)),
    )

    labels = base.mark_text(align="left", dx=10, fontSize=12, color="#8b8a86").encode(
        x=alt.X("Score:Q", scale=x_scale),
        text=alt.Text("Score:Q", format=".4f"),
    )
    return (rule + dots + labels).properties(height=230)

st.set_page_config(page_title="Dry Bean Classifier", page_icon="🫘", layout="wide")

models = load_models()
any_model = next(iter(models.values()))
feature_columns = list(any_model.feature_names_in_)
known_classes = list(any_model.classes_)

st.title("🫘 Dry Bean Variety Classifier")
st.caption(
    "Seven registered dry-bean varieties, identified from 16 shape measurements "
    "taken from grain images (UCI Dry Bean dataset). Five pre-trained models are "
    "scored on the test data you provide; nothing is retrained here."
)

with st.sidebar:
    st.header("1 · Test data")
    upload = st.file_uploader(
        "Upload a test CSV", type="csv",
        help=f"Must contain the 16 feature columns and the '{TARGET_COLUMN}' column.",
    )
    st.download_button(
        "Download sample test_data.csv",
        data=BUNDLED_TEST_FILE.read_bytes(),
        file_name="test_data.csv",
        mime="text/csv",
        help="The 20% hold-out split the models were evaluated on.",
    )

    st.header("2 · Model")
    chosen_model = st.selectbox("Classifier to inspect", list(MODEL_FILES))

if upload is not None:
    try:
        raw = pd.read_csv(upload)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        st.error(f"Could not read the CSV: {exc}")
        st.stop()
    source_label = f"uploaded file **{upload.name}**"
else:
    raw = load_bundled_test_data()
    source_label = "bundled **test_data.csv** (upload your own in the sidebar)"

X_test, y_test, problems = validate_upload(raw, feature_columns, known_classes)
for problem in problems:
    st.warning(problem)
if X_test is None or len(X_test) == 0:
    st.error("No usable rows to evaluate. Check the CSV columns listed above.")
    st.stop()

if y_test.nunique() < len(known_classes):
    st.info("AUC is unavailable because this CSV does not contain all seven varieties. "
            "Macro precision, recall and F1 use the labels present in actual or predicted values.")

st.markdown(
    f"Scoring **{len(X_test):,}** beans from the {source_label} · "
    f"{y_test.nunique()} of {len(known_classes)} varieties present."
)

tab_model, tab_compare, tab_predictions, tab_about = st.tabs(
    ["Selected model", "Compare all five", "Predictions", "About the data"]
)

with tab_model:
    model = models[chosen_model]
    scores = compute_metrics(model, X_test, y_test)

    st.subheader(f"{chosen_model} · evaluation metrics")
    metric_cols = st.columns(len(METRIC_NAMES))
    for col, name in zip(metric_cols, METRIC_NAMES):
        value = scores[name]
        col.metric(name, "n/a" if pd.isna(value) else f"{value:.4f}", help=METRIC_HINTS[name])

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("**Confusion matrix**")
        as_percent = st.toggle("Show as % of each actual variety", value=False)
        st.altair_chart(
            confusion_heatmap(confusion_table(model, X_test, y_test), as_percent),
            width="stretch",
        )
        st.caption("Diagonal = correct. Off-diagonal cells show which varieties get mixed up.")
    with right:
        st.markdown("**Classification report (per variety)**")
        report = per_class_report(model, X_test, y_test)
        st.dataframe(
            report.style.format({"precision": "{:.4f}", "recall": "{:.4f}",
                                 "f1-score": "{:.4f}", "support": "{:.0f}"}),
            width="stretch",
        )

with tab_compare:
    st.subheader("All five models on this test data")
    comparison = pd.DataFrame(
        [{"Model": name, **compute_metrics(m, X_test, y_test)} for name, m in models.items()]
    )

    st.dataframe(
        comparison.style.format({m: "{:.4f}" for m in METRIC_NAMES})
        .highlight_max(subset=METRIC_NAMES, props="font-weight: 700;"),
        hide_index=True, width="stretch",
    )

    metric_to_plot = st.radio("Metric to plot", METRIC_NAMES, index=5, horizontal=True)
    chart = metric_dot_plot(comparison, metric_to_plot, chosen_model)
    if chart is None:
        st.info("This metric is undefined for this data. AUC requires all seven varieties.")
    else:
        st.altair_chart(chart, width="stretch")
        best = comparison.loc[comparison[metric_to_plot].idxmax(), "Model"]
        st.caption(f"Best {metric_to_plot} on this data: **{best}**. "
                   f"The model picked in the sidebar ({chosen_model}) is drawn solid.")

with tab_predictions:
    st.subheader(f"{chosen_model} · individual predictions")
    model = models[chosen_model]
    probabilities = model.predict_proba(X_test)
    predictions = pd.DataFrame({
        "Actual": y_test.to_numpy(),
        "Predicted": model.predict(X_test),
        "Confidence": probabilities.max(axis=1),
    }, index=X_test.index)
    predictions["Correct"] = predictions["Actual"] == predictions["Predicted"]

    only_errors = st.checkbox("Show only misclassified beans", value=True)
    shown = predictions[~predictions["Correct"]] if only_errors else predictions
    n_wrong = int((~predictions["Correct"]).sum())
    st.markdown(f"{n_wrong:,} of {len(predictions):,} beans misclassified.")
    st.dataframe(
        shown.join(X_test).style.format({"Confidence": "{:.2%}"}),
        width="stretch", height=420,
    )
    st.download_button(
        "Download predictions as CSV",
        data=predictions.join(X_test).to_csv(index=False).encode(),
        file_name=f"predictions_{MODEL_FILES[chosen_model].replace('.joblib', '')}.csv",
        mime="text/csv",
    )

with tab_about:
    st.subheader("Dataset")
    st.markdown(
        "- **Source:** UCI Machine Learning Repository - *Dry Bean Dataset* "
        "(Koklu & Ozkan, 2020).\n"
        "- **Size:** 13,611 bean images → 13,543 after removing 68 exact duplicate rows.\n"
        "- **Features (16):** 12 dimensional (Area, Perimeter, axis lengths, "
        "ConvexArea, EquivDiameter, …) and 4 shape factors.\n"
        "- **Target:** 7 varieties - BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA.\n"
        "- **Split:** stratified 80% train / 20% test (seed 42); hyper-parameters "
        "tuned by 5-fold CV on the train part only."
    )
    st.markdown("**Class balance in the scored data**")
    st.dataframe(
        y_test.value_counts().rename_axis("Variety").reset_index(name="Beans"),
        hide_index=True,
    )
    st.markdown("**Reference: metrics recorded at training time (bundled test set)**")
    st.dataframe(
        load_training_metrics().style.format({m: "{:.4f}" for m in METRIC_NAMES}),
        hide_index=True, width="stretch",
    )
