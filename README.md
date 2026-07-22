# Play Store Review Sentiment

Supervised text classification for Google Play Store reviews. The pipeline predicts review polarity (`0` = negative, `1` = positive), compares Naive Bayes and related models, and saves the best full inference pipeline plus an evaluation report.

## What this project does

1. Loads `data/raw/playstore_reviews.csv` (`review`, `polarity` only).
2. Cleans reviews with `strip().str.lower()`.
3. Uses an 80/20 stratified train/test split (`random_state=42`).
4. Trains baseline models on `CountVectorizer(stop_words="english")`:
   - `GaussianNB`, `MultinomialNB`, `BernoulliNB`
   - `RandomForestClassifier`, `LogisticRegression`, `LinearSVC`
5. Runs an enhanced search over Count/TF-IDF features with unigram and bigram ranges.
6. Selects the final model with training-only cross-validated positive-class F1 (accuracy as tie-breaker).
7. Writes model, metrics, and a Markdown evaluation report.

Current selected model: **LinearSVC + TF-IDF unigrams**  
Holdout metrics: accuracy `0.8715`, precision `0.8197`, recall `0.8065`, F1 `0.8130`.

## Project structure

```text
.
├── ai_plan/                          # Spec and implementation plan
├── data/raw/playstore_reviews.csv    # Input dataset
├── models/                           # Saved model + metrics JSON
├── reports/                          # Evaluation report + confusion matrix
├── src/
│   ├── app.py                        # Training, evaluation, persistence
│   ├── explore.ipynb                 # Explanatory notebook
│   └── utils.py                      # Optional database helpers
├── pyproject.toml                    # uv / project dependencies
├── uv.lock                           # Locked dependency versions
└── .python-version                   # Python 3.11
```

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)

## Setup

### Codespaces / Dev Container

The dev container installs dependencies with `uv sync` on create. After the environment is ready:

```bash
uv sync
```

### Local setup

```bash
# Install uv if needed: https://docs.astral.sh/uv/getting-started/installation/
git clone <repo-url>
cd playstore-review-sentiment-nb
uv sync
```

This creates `.venv` and installs the locked packages from `uv.lock`.

## Run the training pipeline

From the repository root:

```bash
uv run python src/app.py
```

This regenerates:

- `models/playstore_review_sentiment_model.joblib` — full vectorizer + classifier pipeline
- `models/playstore_review_sentiment_metrics.json` — metrics and selection details
- `reports/model_evaluation_report.md` — human-readable evaluation report
- `reports/confusion_matrix.png` — final-model confusion matrix

## Explore in the notebook

```bash
uv run jupyter notebook src/explore.ipynb
```

Or open `src/explore.ipynb` in VS Code / Cursor and select the `.venv` kernel.

The notebook reuses functions from `src/app.py` for loading, baseline comparison, enhanced candidate evaluation, and final selection.

## Predict a single review

```bash
uv run python - <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, "src")
from app import predict_review

print(predict_review("this app is amazing and works perfectly"))
print(predict_review("the app keeps crashing and is unusable"))
PY
```

`0` = negative, `1` = positive.

## Dataset notes

- Source: `data/raw/playstore_reviews.csv`
- Features: free-text `review` only (`package_name` is not used)
- Labels: `polarity` in `{0, 1}`
- Class counts in the validated set: 584 negative, 307 positive (891 rows)

## Key design choices

- Vectorizer is fit only on training text (no leakage from the test set).
- `GaussianNB` receives dense arrays; other models keep sparse matrices.
- Enhanced candidates are ranked by mean CV F1 on the training split.
- Holdout accuracy / precision / recall / F1 are reported for diagnostics; they do not drive model selection.
- The saved artifact is a scikit-learn `Pipeline` so preprocessing and inference stay aligned.

## Evaluation report

After training, open:

[`reports/model_evaluation_report.md`](reports/model_evaluation_report.md)

It includes baseline results, enhanced candidate holdout metrics, the selected model, and the confusion matrix.

## Planning docs

- [`ai_plan/playstore_reviews_model_spec.md`](ai_plan/playstore_reviews_model_spec.md)
- [`ai_plan/playstore_reviews_implementation_plan.md`](ai_plan/playstore_reviews_implementation_plan.md)
