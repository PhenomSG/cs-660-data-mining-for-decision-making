# Wine Quality Classification

This project compares three multiclass classifiers on the UCI **red wine quality**
dataset:

- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- Multinomial Logistic Regression

The target is the original wine-quality score (`3` through `8`), not a binary
or grouped target. The same stratified 80/20 split and 5-fold stratified
cross-validation are used for every model. Hyperparameters are selected using
macro-F1 on the training set; the held-out test set is evaluated only after
selection.

## Run

```bash
python wine_quality_models.py
```

The script downloads the UCI archive to `data/` on the first run and writes
metrics and figures to `outputs/`. A local semicolon-delimited CSV can be used
instead:

```bash
python wine_quality_models.py --data path/to/winequality-red.csv
```

## Outputs

- `model_results.csv`: aggregate cross-validation and test metrics
- `per_class_metrics.csv`: precision, recall, F1, and support by quality class
- `*_sensitivity.csv`: every grid-search configuration and CV results
- `quality_distribution.png`
- `correlation_heatmap.png`
- one confusion matrix per model
- `model_comparison.png`

The primary selection metric is macro-F1 because the quality classes are
imbalanced. Accuracy, balanced accuracy, macro/weighted F1, one-vs-rest
ROC-AUC, log loss, and fit time are also reported.

For a complete beginning-to-end explanation suitable for a project review,
see [`STEP_BY_STEP_EXPLANATION.md`](./STEP_BY_STEP_EXPLANATION.md).
