# Play Store Review Sentiment Model Spec

## Project Overview

Build a supervised text classification workflow for `playstore_reviews.csv` to predict whether a mobile application review is negative or positive.

The dataset path is:

```text
/Users/chitrasharathchandra/4geeks/rag_assignments/playstore_reviews.csv
```

Use only these columns:

- `review`: free-text mobile app review
- `polarity`: target label where `0` means negative and `1` means positive

The implementation must train and compare the three Naive Bayes variants requested in the assignment:

- `GaussianNB`
- `MultinomialNB`
- `BernoulliNB`

After comparing the three implementations, choose the best-performing Naive Bayes model, then try to improve or validate the result with a Random Forest model. Finally, train at least one additional studied model that could plausibly outperform Naive Bayes for this use case, explain why it was chosen, and compare its results.

Store the best final model in the appropriate model folder.

## Clarifying Questions

Before implementation, resolve these if possible:

- Should "removing spaces" mean removing leading/trailing whitespace and normalizing repeated spaces, or deleting every space character from each review?
- Should the model selection metric prioritize `accuracy`, `f1_score`, `recall`, or another metric?
- Does the assignment expect a notebook, a Python script, or both?
- What folder convention should be used for stored models: `models/`, `src/models/`, or a course-specific path?
- Should the stored artifact include only the trained estimator, or the full preprocessing/vectorizer/model pipeline?

If no answer is available, use these defaults:

- Treat "removing spaces" as whitespace cleanup: strip leading/trailing spaces and collapse repeated whitespace while preserving token separation.
- Use `accuracy`, `precision`, `recall`, `f1_score`, confusion matrix, and classification report for evaluation.
- Store the full reusable model artifact, including the vectorizer and classifier, in `models/`.

## Tech Stack

- Python 3.10+
- pandas for loading and cleaning the CSV
- scikit-learn for preprocessing, model training, evaluation, and model selection
- joblib for model persistence
- matplotlib or seaborn only if confusion matrix visualization is useful

## Dependencies

Use these Python packages:

```text
pandas
scikit-learn
joblib
matplotlib
seaborn
```

If a dependency file already exists, update it instead of creating a duplicate. If no dependency file exists, create one of the following depending on the project format:

- `requirements.txt` for a simple Python project
- `pyproject.toml` only if the repository already uses that format

## Dataset Handling

Load the CSV with pandas:

```python
df = pd.read_csv("/Users/chitrasharathchandra/4geeks/rag_assignments/playstore_reviews.csv")
```

Required dataset checks:

- Confirm the file loads successfully.
- Confirm `review` and `polarity` exist.
- Keep only `review` and `polarity`.
- Drop rows where `review` or `polarity` is missing.
- Confirm `polarity` only contains `0` and `1`.
- Print or log class distribution before splitting.

Do not use `package_name` for training.

## Preprocessing Requirements

The assignment requires:

1. Removing spaces and converting reviews to lowercase.
2. Splitting the dataset into train and test sets with an 80/20 split.
3. Transforming text using `CountVectorizer` with English stop words and storing transformed data in `X_train` and `X_test`.

Implementation details:

- Convert `review` to string before text cleanup.
- Lowercase the text.
- Normalize whitespace unless the clarification requires deleting all spaces.
- Use `train_test_split` with `test_size=0.2`, `random_state=42`, and `stratify=y`.
- Use `CountVectorizer(stop_words="english")`.

Important: scikit-learn expects `stop_words="english"` in lowercase. Do not pass `"English"` with a capital `E`.

Expected variable flow:

```python
X = df["review"]
y = df["polarity"]

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

vectorizer = CountVectorizer(stop_words="english")
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)
```

## Required Model Training

Train all three Naive Bayes implementations and compare them using the same train/test split and vectorized features.

### GaussianNB

`GaussianNB` does not accept scipy sparse matrices directly. Convert the vectorized matrices to dense arrays only for this model:

```python
gaussian_model.fit(X_train.toarray(), y_train)
gaussian_predictions = gaussian_model.predict(X_test.toarray())
```

### MultinomialNB

`MultinomialNB` is usually the strongest Naive Bayes baseline for count-based text features because word-count features are non-negative frequency counts.

```python
multinomial_model.fit(X_train, y_train)
multinomial_predictions = multinomial_model.predict(X_test)
```

### BernoulliNB

`BernoulliNB` can work for text classification when the feature meaning is word presence/absence rather than word frequency.

```python
bernoulli_model.fit(X_train, y_train)
bernoulli_predictions = bernoulli_model.predict(X_test)
```

## Model Evaluation

For every trained model, report:

- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrix
- Classification report

Use the same evaluation function for every model so results are comparable.

Choose the best Naive Bayes model based primarily on F1 score, with accuracy as a secondary metric. If the dataset is balanced, accuracy can be used as the main metric, but still report F1.

Expected outcome to verify:

- `MultinomialNB` is likely to be the best Naive Bayes choice for `CountVectorizer` features.
- Confirm this empirically instead of assuming it.

## Random Forest Optimization Step

After selecting the best Naive Bayes model, train a `RandomForestClassifier` and compare it against the best Naive Bayes result.

Use the same `X_train`, `X_test`, `y_train`, and `y_test`.

Start with:

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)
```

Then optionally tune:

- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_features`

Use `GridSearchCV` or `RandomizedSearchCV` only if runtime is reasonable. The dataset is small, so cross-validation should be feasible.

Do not claim the Random Forest is better unless its test metrics improve over the selected Naive Bayes model.

## Additional Models to Try

Train at least one additional studied model that could outperform Naive Bayes.

Recommended choices:

- Logistic Regression: often very strong for sparse bag-of-words text classification; handles high-dimensional sparse features well and provides a robust linear baseline.
- Linear Support Vector Machine: usually competitive for text classification because it performs well with high-dimensional sparse vectors and can find a strong separating boundary.
- K-Nearest Neighbors: possible if studied, but usually weaker for sparse text features and should not be the first choice.
- Decision Tree: possible if studied, but usually less stable than linear models or ensembles on sparse text data.

Preferred implementation:

```python
LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42,
)
```

If Linear SVM was covered in the course, also try:

```python
LinearSVC(
    class_weight="balanced",
    random_state=42,
)
```

Argue the model choice in the final notebook/script:

- Naive Bayes assumes conditional independence between words.
- Review sentiment often depends on combinations of words and phrase patterns.
- Logistic Regression and Linear SVM usually handle sparse text features well and can learn more flexible feature weights than Naive Bayes.
- Random Forest can capture nonlinear interactions but may be less efficient on high-dimensional sparse text features.

## Model Persistence

Create a `models/` folder if it does not already exist.

Store the best final artifact with `joblib`.

The stored artifact should include both preprocessing and model inference, preferably as a scikit-learn `Pipeline`:

```python
best_pipeline = Pipeline([
    ("vectorizer", CountVectorizer(stop_words="english")),
    ("model", best_model),
])
```

Fit the final pipeline on the training text data:

```python
best_pipeline.fit(X_train_text, y_train)
joblib.dump(best_pipeline, "models/playstore_review_sentiment_model.joblib")
```

Also store a compact metrics report:

```text
models/playstore_review_sentiment_metrics.json
```

The metrics report should include:

- Model names
- Accuracy, precision, recall, and F1 score for each model
- Selected best model
- Reason the model was selected
- Train/test split settings
- Vectorizer settings

## Development Workflow

1. Inspect the repository structure and reuse any existing folder conventions.
2. Load `playstore_reviews.csv`.
3. Validate required columns and target values.
4. Clean the `review` column.
5. Split the dataset into train and test sets using 80/20.
6. Vectorize the text with `CountVectorizer(stop_words="english")`.
7. Train `GaussianNB`, `MultinomialNB`, and `BernoulliNB`.
8. Evaluate all three models with the same metrics.
9. Select the best Naive Bayes implementation.
10. Train and evaluate a Random Forest model.
11. Train and evaluate at least one additional studied model, preferably Logistic Regression or Linear SVM.
12. Compare all results in a single table.
13. Store the best final pipeline in `models/`.
14. Store metrics and conclusions in a small report or notebook section.

## Constraints

- Use only `review` as the feature and `polarity` as the label.
- Do not use `package_name` as a model feature.
- Keep the train/test split fixed across models.
- Use `random_state=42` for reproducible results.
- Use `stratify=y` in the train/test split.
- Do not fit the vectorizer on the full dataset before splitting.
- Fit the vectorizer only on `X_train_text`, then transform `X_test_text`.
- Avoid data leakage from the test set.
- Use `stop_words="english"` exactly for `CountVectorizer`.
- Convert sparse matrices to dense only for `GaussianNB`.
- Do not overwrite stored models without confirming the intended filename if a model artifact already exists.

## Additional Tasks to Improve Model Outcomes

Suggest these to the coding agent as optional improvements after the required assignment is complete:

- Compare `CountVectorizer` against `TfidfVectorizer`.
- Try `ngram_range=(1, 2)` to capture short sentiment phrases.
- Tune `min_df` and `max_df` to remove very rare or overly common tokens.
- Inspect misclassified reviews to identify preprocessing issues.
- Check class balance and consider class weighting where supported.
- Use cross-validation to reduce dependence on a single split.
- Save the final model as a full pipeline to prevent training/inference mismatch.
- Add a small prediction function that accepts a raw review string and returns the predicted polarity.

## Expected Deliverables

The coding agent should produce:

- A working notebook or Python script that trains and evaluates the models.
- A comparison table for `GaussianNB`, `MultinomialNB`, `BernoulliNB`, Random Forest, and the additional studied model.
- A short written conclusion explaining which model won and why.
- A saved model artifact in `models/playstore_review_sentiment_model.joblib`.
- A saved metrics report in `models/playstore_review_sentiment_metrics.json`.

