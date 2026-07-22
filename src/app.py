"""Play Store review sentiment classification pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "playstore_reviews.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_PATH = MODELS_DIR / "playstore_review_sentiment_model.joblib"
METRICS_PATH = MODELS_DIR / "playstore_review_sentiment_metrics.json"
REPORT_PATH = REPORTS_DIR / "model_evaluation_report.md"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"

RANDOM_STATE = 42
TEST_SIZE = 0.2
POS_LABEL = 1
CV_FOLDS = 5


@dataclass
class ModelMetrics:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list[list[int]]
    classification_report: dict[str, Any]
    evaluation_type: str = "holdout"


def load_and_validate_data(csv_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load CSV and keep only validated review/polarity rows."""
    df = pd.read_csv(csv_path)
    required_columns = {"review", "polarity"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df = df[["review", "polarity"]].copy()
    df["review"] = df["review"].astype(str)
    df = df.dropna(subset=["review", "polarity"])
    df = df[df["review"].str.strip().ne("")]
    df["polarity"] = df["polarity"].astype(int)

    invalid_labels = set(df["polarity"].unique()) - {0, 1}
    if invalid_labels:
        raise ValueError(f"polarity must contain only 0 and 1, found: {sorted(invalid_labels)}")

    return df.reset_index(drop=True)


def preprocess_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace and lowercase review text."""
    cleaned = df.copy()
    cleaned["review"] = cleaned["review"].str.strip().str.lower()
    return cleaned


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Create a stratified train/test split."""
    x = df["review"]
    y = df["polarity"]
    return train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def evaluate_predictions(
    model_name: str,
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    evaluation_type: str = "holdout",
) -> ModelMetrics:
    """Compute comparable metrics for one model."""
    return ModelMetrics(
        model_name=model_name,
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0)),
        confusion_matrix=confusion_matrix(y_true, y_pred).tolist(),
        classification_report=classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
        evaluation_type=evaluation_type,
    )


def _build_count_vectorizer() -> CountVectorizer:
    return CountVectorizer(stop_words="english")


def train_baseline_models(
    x_train_text: pd.Series,
    x_test_text: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[list[ModelMetrics], dict[str, Any], str]:
    """Train required baseline models on CountVectorizer features."""
    vectorizer = _build_count_vectorizer()
    x_train = vectorizer.fit_transform(x_train_text)
    x_test = vectorizer.transform(x_test_text)

    baseline_models: dict[str, Any] = {
        "GaussianNB": GaussianNB(),
        "MultinomialNB": MultinomialNB(),
        "BernoulliNB": BernoulliNB(),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "LinearSVC": LinearSVC(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }

    metrics: list[ModelMetrics] = []
    fitted_models: dict[str, Any] = {}

    for name, model in baseline_models.items():
        if name == "GaussianNB":
            model.fit(x_train.toarray(), y_train)
            predictions = model.predict(x_test.toarray())
        else:
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)

        metrics.append(evaluate_predictions(name, y_test, predictions, evaluation_type="holdout"))
        fitted_models[name] = model

    naive_bayes_names = {"GaussianNB", "MultinomialNB", "BernoulliNB"}
    best_nb = max(
        (metric for metric in metrics if metric.model_name in naive_bayes_names),
        key=lambda metric: (metric.f1, metric.accuracy),
    )

    return metrics, fitted_models, best_nb.model_name


def _candidate_pipelines() -> list[tuple[str, Pipeline, dict[str, list[Any]]]]:
    """Build leakage-safe pipelines for enhanced cross-validation."""
    candidates: list[tuple[str, Pipeline, dict[str, list[Any]]]] = []

    vectorizers = {
        "count_unigram": CountVectorizer(stop_words="english"),
        "count_bigram": CountVectorizer(stop_words="english", ngram_range=(1, 2)),
        "tfidf_unigram": TfidfVectorizer(stop_words="english"),
        "tfidf_bigram": TfidfVectorizer(stop_words="english", ngram_range=(1, 2)),
    }

    model_specs = {
        "MultinomialNB": (
            MultinomialNB(),
            {"model__alpha": [0.1, 1.0, 5.0]},
        ),
        "BernoulliNB": (
            BernoulliNB(),
            {"model__alpha": [0.1, 1.0, 5.0]},
        ),
        "LogisticRegression": (
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        "LinearSVC": (
            LinearSVC(class_weight="balanced", random_state=RANDOM_STATE),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        "RandomForestClassifier": (
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,
            ),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 20],
                "model__min_samples_split": [2, 5],
            },
        ),
    }

    for vectorizer_name, vectorizer in vectorizers.items():
        for model_name, (model, param_grid) in model_specs.items():
            pipeline = Pipeline(
                [
                    ("vectorizer", clone(vectorizer)),
                    ("model", clone(model)),
                ]
            )
            candidates.append(
                (
                    f"{model_name}__{vectorizer_name}",
                    pipeline,
                    param_grid,
                )
            )

    return candidates


def run_enhanced_cv(
    x_train_text: pd.Series,
    y_train: pd.Series,
    x_test_text: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Tune candidates with CV and report each tuned candidate on holdout data."""
    scoring = {
        "accuracy": "accuracy",
        "f1": make_scorer(f1_score, zero_division=0),
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
    }
    cv_rows: list[dict[str, Any]] = []
    best_candidate: dict[str, Any] | None = None

    for candidate_name, pipeline, param_grid in _candidate_pipelines():
        search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            scoring=scoring,
            cv=CV_FOLDS,
            n_jobs=-1,
            refit="f1",
        )
        search.fit(x_train_text, y_train)

        best_index = search.best_index_
        holdout_metrics = evaluate_predictions(
            candidate_name,
            y_test,
            search.best_estimator_.predict(x_test_text),
            evaluation_type="holdout_diagnostic",
        )

        row = {
            "candidate": candidate_name,
            "best_params": search.best_params_,
            "cv_accuracy_mean": float(search.cv_results_["mean_test_accuracy"][best_index]),
            "cv_precision_mean": float(search.cv_results_["mean_test_precision"][best_index]),
            "cv_recall_mean": float(search.cv_results_["mean_test_recall"][best_index]),
            "cv_f1_mean": float(search.cv_results_["mean_test_f1"][best_index]),
            "holdout_accuracy": holdout_metrics.accuracy,
            "holdout_precision": holdout_metrics.precision,
            "holdout_recall": holdout_metrics.recall,
            "holdout_f1": holdout_metrics.f1,
        }
        cv_rows.append(row)

        if best_candidate is None or (
            row["cv_f1_mean"],
            row["cv_accuracy_mean"],
        ) > (
            best_candidate["cv_f1_mean"],
            best_candidate["cv_accuracy_mean"],
        ):
            best_candidate = {
                **row,
                "pipeline": search.best_estimator_,
            }

    if best_candidate is None:
        raise RuntimeError("No enhanced CV candidates were evaluated.")

    cv_df = pd.DataFrame(cv_rows).sort_values(
        by=["cv_f1_mean", "cv_accuracy_mean"],
        ascending=False,
    )
    return cv_df, best_candidate


def select_final_model(
    best_cv_candidate: dict[str, Any],
    x_test_text: pd.Series,
    y_test: pd.Series,
) -> tuple[Pipeline, ModelMetrics, str]:
    """Select by training-only CV rank and report final holdout metrics."""
    cv_pipeline: Pipeline = best_cv_candidate["pipeline"]
    cv_holdout = evaluate_predictions(
        best_cv_candidate["candidate"],
        y_test,
        cv_pipeline.predict(x_test_text),
        evaluation_type="holdout_final",
    )

    return cv_pipeline, cv_holdout, (
        f"Selected {cv_holdout.model_name} using the highest mean training-set CV "
        f"positive-class F1 ({best_cv_candidate['cv_f1_mean']:.4f}), with CV accuracy "
        f"as the tie-breaker. Its holdout F1 ({cv_holdout.f1:.4f}) is reported as "
        "an out-of-sample evaluation and was not used for selection."
    )


def save_confusion_matrix_plot(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    output_path: Path = CONFUSION_MATRIX_PATH,
) -> None:
    """Save a confusion matrix figure for the final model."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative (0)", "Positive (1)"],
        yticklabels=["Negative (0)", "Positive (1)"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Final Model Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def metrics_to_dict(metrics: list[ModelMetrics]) -> list[dict[str, Any]]:
    """Convert metrics dataclasses to JSON-safe dictionaries."""
    return [asdict(metric) for metric in metrics]


def write_metrics_json(payload: dict[str, Any], output_path: Path = METRICS_PATH) -> None:
    """Persist metrics as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def collect_misclassified_examples(
    x_test_text: pd.Series,
    y_test: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, list[dict[str, Any]]]:
    """Collect false-positive and false-negative holdout reviews."""
    results = pd.DataFrame(
        {
            "review": x_test_text.to_numpy(),
            "actual": np.asarray(y_test),
            "predicted": np.asarray(y_pred),
        }
    )
    false_positives = results[(results["actual"] == 0) & (results["predicted"] == 1)]
    false_negatives = results[(results["actual"] == 1) & (results["predicted"] == 0)]
    return {
        "false_positives": false_positives.to_dict(orient="records"),
        "false_negatives": false_negatives.to_dict(orient="records"),
    }


def _format_misclassified_section(title: str, examples: list[dict[str, Any]]) -> str:
    """Format misclassified reviews as a Markdown section."""
    lines = [f"### {title} ({len(examples)})", ""]
    if not examples:
        lines.append("_None._")
        lines.append("")
        return "\n".join(lines)

    for index, example in enumerate(examples, start=1):
        review = str(example["review"]).replace("\n", " ").strip()
        lines.append(f"{index}. {review}")
        lines.append("")
    return "\n".join(lines)


def write_markdown_report(payload: dict[str, Any], output_path: Path = REPORT_PATH) -> None:
    """Generate a human-readable evaluation report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    baseline_rows = payload["baseline_metrics"]
    cv_rows = payload["enhanced_cv_results"]
    final_metrics = payload["final_model_metrics"]
    dataset = payload["dataset_summary"]

    baseline_table_lines = [
        "| Model | Accuracy | Precision | Recall | F1 (pos) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(
        baseline_rows,
        key=lambda row: (row["accuracy"], row["f1"]),
        reverse=True,
    ):
        baseline_table_lines.append(
            f"| {row['model_name']} | {row['accuracy']:.4f} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['f1']:.4f} |"
        )

    candidate_table_lines = [
        "| Candidate | Holdout Accuracy | Holdout Precision | Holdout Recall | Holdout F1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    holdout_rows = sorted(
        cv_rows,
        key=lambda row: (row["holdout_accuracy"], row["holdout_f1"]),
        reverse=True,
    )
    for row in holdout_rows:
        candidate_table_lines.append(
            f"| {row['candidate']} | {row['holdout_accuracy']:.4f} | "
            f"{row['holdout_precision']:.4f} | "
            f"{row['holdout_recall']:.4f} | {row['holdout_f1']:.4f} |"
        )
    highest_holdout_accuracy = holdout_rows[0]

    misclassified = payload.get(
        "misclassified_examples",
        {"false_positives": [], "false_negatives": []},
    )
    false_positive_section = _format_misclassified_section(
        "False positives — labeled negative (0), predicted positive (1)",
        misclassified.get("false_positives", []),
    )
    false_negative_section = _format_misclassified_section(
        "False negatives — labeled positive (1), predicted negative (0)",
        misclassified.get("false_negatives", []),
    )

    report = f"""# Play Store Review Sentiment Model Evaluation Report

## Dataset Summary

- Total valid reviews: {dataset['total_rows']}
- Negative reviews (0): {dataset['class_counts']['0']}
- Positive reviews (1): {dataset['class_counts']['1']}
- Source file: `{dataset['source_path']}`

## Preprocessing and Split Settings

- Review cleaning: `df["review"] = df["review"].str.strip().str.lower()`
- Train/test split: 80/20 stratified split
- Random state: {payload['split_settings']['random_state']}
- Primary selection metric: positive-class F1 (`pos_label=1`)
- Tie-breaker: accuracy

## Baseline Holdout Results (CountVectorizer, unigrams)

{chr(10).join(baseline_table_lines)}

- Best Naive Bayes model: **{payload['best_naive_bayes']}**

## Enhanced Candidate Results

Vectorizer and model combinations were tuned with `{CV_FOLDS}`-fold stratified cross-validation on the training split only.
The table shows only holdout accuracy, precision, recall, and F1 for each tuned candidate.
Cross-validation remains internal to model selection; for readability, this diagnostic table is ordered by holdout accuracy and then holdout F1.

{chr(10).join(candidate_table_lines)}

- Best CV candidate: **{payload['best_cv_candidate']['candidate']}**
- Best CV params: `{payload['best_cv_candidate']['best_params']}`
- Highest diagnostic holdout accuracy: **{highest_holdout_accuracy['candidate']}** ({highest_holdout_accuracy['holdout_accuracy']:.4f})

## Final Selected Model

- Selected model: **{final_metrics['model_name']}**
- Selection rationale: {payload['selection_rationale']}
- Holdout accuracy: {final_metrics['accuracy']:.4f}
- Holdout precision (positive): {final_metrics['precision']:.4f}
- Holdout recall (positive): {final_metrics['recall']:.4f}
- Holdout F1 (positive): {final_metrics['f1']:.4f}

### Final Holdout Confusion Matrix

![Confusion Matrix](confusion_matrix.png)

```
{final_metrics['confusion_matrix']}
```

### Final Classification Report

```json
{json.dumps(final_metrics['classification_report'], indent=2)}
```

## Misclassified Holdout Reviews

From the confusion matrix above: **{len(misclassified.get('false_positives', []))} false positives** and **{len(misclassified.get('false_negatives', []))} false negatives**.

Many errors are mixed-sentiment reviews (praise plus complaints) or borderline labels.

{false_positive_section}
{false_negative_section}
## Conclusion

The final model was selected by mean positive-class F1 from training-only cross-validation, with CV accuracy as a tie-breaker. Among the Naive Bayes variants, **{payload['best_naive_bayes']}** performed best on the required baseline comparison. After enhanced tuning with Count/TF-IDF features and unigram/bigram ranges, **{final_metrics['model_name']}** was chosen as the production artifact. Its holdout metrics estimate out-of-sample performance; the per-candidate holdout metrics above are diagnostic comparisons and did not control model selection.
"""

    with output_path.open("w", encoding="utf-8") as file:
        file.write(report)


def save_model_pipeline(
    pipeline: Pipeline,
    output_path: Path = MODEL_PATH,
    overwrite: bool = False,
) -> None:
    """Persist the final fitted pipeline."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Model artifact already exists at {output_path}. "
            "Pass overwrite=True to replace it."
        )
    joblib.dump(pipeline, output_path)


def predict_review(review_text: str, model_path: Path = MODEL_PATH) -> int:
    """Predict polarity for a single raw review string."""
    pipeline: Pipeline = joblib.load(model_path)
    cleaned = pd.Series([str(review_text)]).str.strip().str.lower()
    return int(pipeline.predict(cleaned)[0])


def run_training_pipeline(
    csv_path: Path = DATA_PATH,
    overwrite_artifacts: bool = True,
) -> dict[str, Any]:
    """Execute the full training, evaluation, and persistence workflow."""
    raw_df = load_and_validate_data(csv_path)
    df = preprocess_reviews(raw_df)
    x_train_text, x_test_text, y_train, y_test = split_data(df)

    baseline_metrics, _, best_nb_name = train_baseline_models(
        x_train_text,
        x_test_text,
        y_train,
        y_test,
    )
    cv_df, best_cv_candidate = run_enhanced_cv(
        x_train_text,
        y_train,
        x_test_text,
        y_test,
    )
    final_pipeline, final_metrics, selection_rationale = select_final_model(
        best_cv_candidate,
        x_test_text,
        y_test,
    )

    final_pipeline.fit(x_train_text, y_train)
    final_predictions = final_pipeline.predict(x_test_text)
    save_confusion_matrix_plot(y_test, final_predictions)
    misclassified_examples = collect_misclassified_examples(
        x_test_text,
        y_test,
        final_predictions,
    )

    class_counts = df["polarity"].value_counts().sort_index()
    payload: dict[str, Any] = {
        "dataset_summary": {
            "total_rows": int(len(df)),
            "class_counts": {str(k): int(v) for k, v in class_counts.items()},
            "source_path": str(csv_path.relative_to(PROJECT_ROOT)),
        },
        "split_settings": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratify": True,
            "primary_metric": "f1_pos",
            "tie_breaker": "accuracy",
        },
        "vectorizer_settings": {
            "baseline": {"type": "CountVectorizer", "stop_words": "english", "ngram_range": [1, 1]},
            "enhanced": "Count/TF-IDF with unigram and bigram ranges",
            "cv_folds": CV_FOLDS,
        },
        "baseline_metrics": metrics_to_dict(baseline_metrics),
        "best_naive_bayes": best_nb_name,
        "enhanced_cv_results": cv_df.to_dict(orient="records"),
        "best_cv_candidate": {
            "candidate": best_cv_candidate["candidate"],
            "best_params": best_cv_candidate["best_params"],
            "cv_accuracy_mean": best_cv_candidate["cv_accuracy_mean"],
            "cv_precision_mean": best_cv_candidate["cv_precision_mean"],
            "cv_recall_mean": best_cv_candidate["cv_recall_mean"],
            "cv_f1_mean": best_cv_candidate["cv_f1_mean"],
        },
        "final_model_metrics": asdict(final_metrics),
        "selection_rationale": selection_rationale,
        "misclassified_examples": misclassified_examples,
        "artifact_paths": {
            "model": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
            "metrics_json": str(METRICS_PATH.relative_to(PROJECT_ROOT)),
            "report_markdown": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
            "confusion_matrix_png": str(CONFUSION_MATRIX_PATH.relative_to(PROJECT_ROOT)),
        },
    }

    save_model_pipeline(final_pipeline, overwrite=overwrite_artifacts)
    write_metrics_json(payload)
    write_markdown_report(payload)

    return payload


def main() -> None:
    """Run the end-to-end training pipeline."""
    payload = run_training_pipeline()
    final_metrics = payload["final_model_metrics"]
    print("Training complete.")
    print(f"Best Naive Bayes: {payload['best_naive_bayes']}")
    print(f"Selected model: {final_metrics['model_name']}")
    print(f"Holdout F1 (positive): {final_metrics['f1']:.4f}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
