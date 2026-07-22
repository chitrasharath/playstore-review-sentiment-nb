# Play Store Review Sentiment Implementation Plan

## Overview

Build a reproducible notebook-and-script sentiment-classification workflow using `data/raw/playstore_reviews.csv`, compare all required and enhanced models without data leakage, and persist the winning inference pipeline plus metrics.

Decisions confirmed with the user:

- Dataset location: `data/raw/playstore_reviews.csv`
- Deliverables: develop in `src/explore.ipynb`, then migrate clean code to `src/app.py`
- "Removing spaces" is implemented exactly as `df["review"] = df["review"].str.strip().str.lower()` (strip leading/trailing whitespace, lowercase; internal whitespace untouched)
- Selection metric: positive-class F1, with accuracy as tie-breaker
- Additional models: Logistic Regression and Linear SVM
- Enhanced scope: also test TF-IDF, bigrams, and cross-validation
- Environment/tooling: manage the project with `uv` (full `pyproject.toml` + `uv.lock`), migrate and remove `requirements.txt`, and update the devcontainer to install and use `uv`

## Environment and tooling (uv)

- Initialize a full uv-managed project with a `pyproject.toml` and committed `uv.lock`, targeting Python 3.11 (current interpreter is 3.11.4) and pinning it via `.python-version`.
- Migrate every package currently in `requirements.txt` (`ipyleaflet`, `ipywidgets`, `matplotlib`, `numpy`, `opencv-python`, `pandas`, `psycopg2-binary`, `pymysql`, `python-dotenv`, `requests`, `scikit-learn`, `seaborn`, `sqlalchemy`, `sympy`, `xgboost`) into `pyproject.toml` dependencies, add `joblib` explicitly, then delete `requirements.txt`.
- Add `ipykernel` so the notebook runs inside the uv environment; resolve and lock everything with `uv sync`.
- Update the devcontainer to provision uv: add the `ghcr.io/va-h/devcontainers-features/uv:1` feature in `.devcontainer/devcontainer.json` and change `postCreateCommand` from `pip install -r requirements.txt` to `uv sync`.
- Run all code through the uv environment (`uv run python src/app.py`, `uv run` for the notebook kernel) so the notebook and script share one locked environment.

## Deliverables and project structure

- Create `pyproject.toml` + `uv.lock` as the single source of dependency truth (replacing `requirements.txt`), including an explicit `joblib` dependency.
- Replace the boilerplate in `src/app.py` with reusable data validation, preprocessing, training, evaluation, selection, persistence, and prediction functions plus a runnable `main()`.
- Build the explanatory workflow in `src/explore.ipynb`, calling the reusable script functions where practical so notebook and production logic do not drift.
- Read `data/raw/playstore_reviews.csv` via a repository-relative path and write the final pipeline and report to `models/playstore_review_sentiment_model.joblib` and `models/playstore_review_sentiment_metrics.json`.

## Data preparation and leakage controls

- Validate the required `review` and `polarity` columns, retain only those columns, remove missing/empty rows, enforce binary integer labels, and report class counts (currently 584 negative and 307 positive across 891 valid rows).
- Clean reviews by converting to strings, then applying `df["review"] = df["review"].str.strip().str.lower()` — strip leading/trailing whitespace and lowercase, leaving internal spacing intact.
- Create one stratified 80/20 holdout split with `random_state=42`; fit every vectorizer and all tuning folds only on training text.
- Use positive-class F1 (`pos_label=1`) as the primary selection metric and accuracy as the tie-breaker; also report precision, recall, confusion matrices, and classification reports.

## Required baseline comparison

- Fit one `CountVectorizer(stop_words="english")` on training text and reuse its matrices for `GaussianNB`, `MultinomialNB`, `BernoulliNB`, and the initial `RandomForestClassifier` comparison.
- Densify only the Gaussian Naive Bayes inputs; keep all other model inputs sparse.
- Evaluate each model through one shared metrics function and explicitly identify the strongest Naive Bayes model before comparing Random Forest.
- Add Logistic Regression and Linear SVM as the two requested high-dimensional sparse-text alternatives.

## Enhanced model selection

- Use stratified cross-validation on training data to compare leakage-safe pipelines with Count versus TF-IDF features, unigram versus unigram/bigram ranges, and reasonable model-specific parameters.
- Keep the search bounded for this 891-row dataset: tune Naive Bayes smoothing, Logistic Regression regularization, Linear SVM regularization, and a compact Random Forest parameter space.
- Rank candidates by mean cross-validated positive-class F1, break practical ties with cross-validated accuracy, then evaluate the selected configuration once on the untouched test set.
- Refit the selected full vectorizer/model pipeline on the original training split, preserving the specified holdout semantics, and serialize that complete inference pipeline. Expose a helper that accepts raw review text and returns polarity.

## Notebook narrative and artifacts

- Organize the notebook into dataset checks, class distribution, preprocessing, baseline results, enhanced cross-validation, final holdout comparison, confusion matrix, and a short evidence-based conclusion.
- Produce a single comparison DataFrame covering all required models plus Logistic Regression and Linear SVM, clearly separating baseline holdout metrics from cross-validation results.
- Save JSON-safe metrics containing all model scores, best Naive Bayes, final selected model and rationale, split settings, vectorizer settings, CV setup, confusion matrix, and classification report.
- Refuse to overwrite an existing final artifact unless explicitly allowed; the current `models/` directory contains only `.gitkeep`.

## Evaluation report

- Generate a human-readable Markdown report at `reports/model_evaluation_report.md` (creating the `reports/` folder) that summarizes evaluation and outcomes, written programmatically by `src/app.py` from the same metrics used for the JSON artifact so the two never diverge.
- The report must include: dataset summary and class distribution, preprocessing and split settings, vectorizer/CV configuration, a comparison table of all models (GaussianNB, MultinomialNB, BernoulliNB, Random Forest, Logistic Regression, Linear SVM) with accuracy, precision, recall, and positive-class F1, the best Naive Bayes model, the final selected model with its rationale, the final holdout confusion matrix and classification report, and a short written conclusion on which model won and why.
- Embed the saved confusion-matrix figure in the report and keep its numbers consistent with `models/playstore_review_sentiment_metrics.json`.

## Verification

- Confirm `uv sync` resolves cleanly and produces `uv.lock`; verify `requirements.txt` is gone and the devcontainer uses `uv`.
- Run `uv run python src/app.py` from the repository root and confirm deterministic completion, valid model/report artifacts, and no data leakage warnings or dense conversion outside GaussianNB.
- Load the saved joblib pipeline in a fresh process and smoke-test predictions for representative positive and negative raw reviews.
- Execute all notebook cells from a clean kernel and verify that its reported winner and metrics agree with the saved JSON report.
- Confirm `reports/model_evaluation_report.md` is generated, renders correctly, and its comparison table, selected model, and confusion matrix match the JSON metrics.

## Implementation steps

1. Save this plan in `ai_plan/`, initialize the uv project (`pyproject.toml`, `.python-version`, `uv.lock`), migrate dependencies plus `joblib`/`ipykernel`, remove `requirements.txt`, and update the devcontainer to install and use uv.
2. Implement validated, reusable training and evaluation logic in `src/app.py`.
3. Create the complete explanatory analysis in `src/explore.ipynb` using the shared logic.
4. Run baseline and enhanced comparisons, persist the winner and metrics, generate `reports/model_evaluation_report.md`, then verify the script, notebook, report, and inference artifacts.
