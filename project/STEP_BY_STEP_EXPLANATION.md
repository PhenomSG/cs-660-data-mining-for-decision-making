# Step-by-Step Explanation for the Wine Quality Experiment

This document explains the implementation from the beginning to the end in
the form that can be used during a project review or viva.

## 1. What is the project trying to do?

The objective is to classify the quality of red wine using its chemical
properties. Three classification algorithms are compared:

1. K-Nearest Neighbors (KNN)
2. Gaussian Naive Bayes
3. Multinomial Logistic Regression

The important part of the assignment is **sensitivity analysis**. Sensitivity
analysis means changing one or more algorithm parameters, measuring the
performance for each configuration, and identifying which configuration
generalizes best.

The implementation is in
[`wine_quality_models.py`](./wine_quality_models.py).

## Simple explanation of sensitivity analysis

Sensitivity analysis means checking how much a model's performance changes when
we change its important parameters.

In this project, we did not train only one version of each algorithm. Instead,
we trained several versions of KNN, Gaussian Naive Bayes, and Logistic
Regression. Each version used a different parameter combination. We then
compared their validation macro-F1 scores.

For example, for KNN we tested different values of `n_neighbors`. If the
values are 3, 5, and 7, then we train and evaluate KNN three times with
different neighborhood sizes. This tells us whether the model works better
with a small or large neighborhood.

The purpose is to:

1. Find parameter values that give the best generalization performance.
2. Understand which parameters strongly affect the model.
3. Detect underfitting and overfitting.
4. Avoid choosing a model based on one arbitrary parameter setting.
5. Provide evidence for why the final configuration was selected.

In this implementation, sensitivity analysis is performed by
`GridSearchCV`. It evaluates every parameter combination using five-fold
stratified cross-validation. The primary comparison measure is macro-F1,
because the wine-quality classes are imbalanced.

The parameter values are **not changed during test-set evaluation**. They are
chosen using the training data and cross-validation first. The untouched test
set is used only at the end to estimate final performance.

## Parameters analysed in this project

### KNN parameters

KNN tested three parameters:

| Parameter | Values tested | What it controls |
|---|---|---|
| `n_neighbors` | 3, 5, 7, 9, 11, 15, 21 | Number of nearby wines used to vote |
| `weights` | `uniform`, `distance` | Whether every neighbor has equal influence or closer neighbors have more influence |
| `metric` | `euclidean`, `manhattan` | How distance between two wines is calculated |

This gives:

```text
7 neighbor values × 2 weighting options × 2 distance metrics = 28 configurations
```

The selected KNN configuration was:

```text
n_neighbors = 5
weights = distance
metric = euclidean
```

What this means: using five nearby wines, giving closer wines more influence,
and measuring distance using the Euclidean formula produced the best mean
cross-validation macro-F1 among the tested KNN configurations.

### Gaussian Naive Bayes parameter

Gaussian Naive Bayes tested:

| Parameter | Values tested | What it controls |
|---|---|---|
| `var_smoothing` | `1e-11`, `1e-10`, `1e-9`, `1e-8`, `1e-7`, `1e-6`, `1e-5` | Adds a small stabilizing value to feature variances |

The selected configuration was:

```text
var_smoothing = 1e-11
```

In this experiment, all tested smoothing values produced the same
cross-validation score to the displayed precision. The first tied
configuration was retained as the best result.

What this means: the model's performance was not sensitive to the tested
smoothing range for this dataset. The parameter was still analysed because
variance smoothing can prevent unstable probability calculations when a
feature has a very small variance.

### Logistic Regression parameters

Logistic Regression tested two parameters:

| Parameter | Values tested | What it controls |
|---|---|---|
| `C` | 0.001, 0.01, 0.1, 1, 10, 100 | Inverse regularization strength |
| `class_weight` | `None`, `balanced` | Whether minority classes receive extra importance |

This gives:

```text
6 C values × 2 class-weight options = 12 configurations
```

The selected configuration was:

```text
C = 10
class_weight = None
```

What this means: among the tested settings, a less strongly regularized model
with the original class frequencies produced the highest mean validation
macro-F1.

## What does changing a parameter do?

Changing a parameter changes the model's behaviour before it predicts the
test examples:

- In KNN, it changes which neighboring wines influence a prediction.
- In Gaussian Naive Bayes, it changes the numerical smoothing applied to
  feature variances.
- In Logistic Regression, it changes how strongly model coefficients are
  constrained and whether minority classes receive additional weight.

For every setting, we record a performance score. We then select the setting
with the best mean validation macro-F1. Thus, sensitivity analysis converts
parameter selection from a guess into a measured comparison.

## Where are the different results shown?

Every tested configuration is stored in:

```text
outputs/knn_sensitivity.csv
outputs/gaussian_naive_bayes_sensitivity.csv
outputs/logistic_regression_sensitivity.csv
```

Important columns in these files are:

| Column | Meaning |
|---|---|
| `params` | Exact parameter combination tested |
| `mean_test_score` | Mean validation macro-F1 across five folds |
| `std_test_score` | Variation of the validation score across folds |
| `rank_test_score` | Rank of the configuration; rank 1 is best |
| `mean_train_score` | Mean training macro-F1 |

The selected parameter combination and final test results are summarized in:

```text
outputs/model_results.csv
```

## Short answer for the instructor

> Sensitivity analysis is the systematic study of how changing an algorithm's
> parameters affects its performance. In our project, we used `GridSearchCV`
> with five-fold stratified cross-validation. For KNN, we varied the number of
> neighbors, the neighbor weighting method, and the distance metric. For
> Gaussian Naive Bayes, we varied variance smoothing. For Logistic Regression,
> we varied regularization strength `C` and class weighting. Every parameter
> combination was measured using mean validation macro-F1. We selected the
> combination with the highest macro-F1, saved all combinations in the
> sensitivity CSV files, and evaluated the selected models once on the
> untouched test set.

## 2. Which dataset is used?

The experiment uses the **UCI Red Wine Quality dataset**:

<https://archive.ics.uci.edu/dataset/186/wine+quality>

The script downloads the official UCI ZIP file automatically if the data is
not already present. The extracted file is:

```text
data/winequality-red.csv
```

The dataset contains:

- 1,599 wine observations
- 11 numerical input features
- One target column called `quality`
- Quality labels from 3 to 8

The input features are:

```text
fixed acidity
volatile acidity
citric acid
residual sugar
chlorides
free sulfur dioxide
total sulfur dioxide
density
pH
sulphates
alcohol
```

The target is the original quality score. We do not convert it into low,
medium, and high categories in this experiment.

## 3. Why is this a multiclass classification problem?

The target is not continuous in this experiment. It is treated as a set of
classes:

```text
3, 4, 5, 6, 7, 8
```

Therefore, the model predicts which quality class a wine belongs to. This is
multiclass classification because there are more than two possible classes.

The classes are imbalanced. Most wines have quality 5 or 6, while qualities 3
and 8 have very few observations. This is why accuracy alone is not enough.

## 4. Initial data processing

The following steps happen before training:

1. Load the CSV using semicolon separation.
2. Check that the `quality` column exists.
3. Check for missing values.
4. Separate features and target:

```python
X = data.drop(columns="quality")
y = data["quality"]
```

5. Save exploratory visualizations:
   - `quality_distribution.png`
   - `correlation_heatmap.png`

The quality-distribution plot shows class imbalance. The correlation heatmap
shows relationships between chemical features.

## 5. How is the data split?

The data is divided into:

- 80% training data
- 20% final test data

The split is stratified:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)
```

### Why use stratification?

Stratification preserves approximately the same proportion of each quality
class in both the training and test sets. This is important because the target
classes are imbalanced.

### Why use `random_state=42`?

It makes the split reproducible. If we run the script again, we get the same
training and test observations.

## 6. Why is StandardScaler used?

The chemical features have different numerical ranges. For example, alcohol,
pH, density, and sulfur dioxide are measured on different scales.

The implementation uses:

```python
StandardScaler()
```

The scaler transforms each feature approximately to:

```text
mean = 0
standard deviation = 1
```

Scaling is especially important for KNN because KNN uses distances. It is also
useful for Logistic Regression. It is not mathematically required for Gaussian
Naive Bayes, but using the same safe preprocessing structure makes the
comparison consistent.

The scaler is inside a scikit-learn `Pipeline`:

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", model),
])
```

This prevents data leakage. During cross-validation, the scaler is fitted
only on the training portion of each fold, not on the validation portion.

## 7. How is sensitivity analysis implemented?

Sensitivity analysis is implemented in two places:

1. Parameter grids in `build_models()`
2. `GridSearchCV` in `run_experiment()`

The key code pattern is:

```python
search = GridSearchCV(
    pipeline,
    parameter_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    return_train_score=True,
)
search.fit(X_train, y_train)
```

The parameter grid contains the values to test. `GridSearchCV` tests every
combination of those values.

The evaluation uses:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)
```

Therefore, each parameter configuration is evaluated on five validation
folds. The five validation scores are averaged.

### What is the selection metric?

The primary selection metric is **macro-F1**:

```python
scoring="f1_macro"
```

Macro-F1 calculates F1 separately for every quality class and then gives each
class equal weight. This is more appropriate than optimizing only accuracy
because quality 3 and quality 8 are rare.

The best configuration is the one with the highest mean validation
macro-F1, not the highest training score.

## 8. Where can the sensitivity results be seen?

The script writes one sensitivity file for each algorithm:

```text
outputs/knn_sensitivity.csv
outputs/gaussian_naive_bayes_sensitivity.csv
outputs/logistic_regression_sensitivity.csv
```

These files contain:

- The tested parameter values
- The parameter combination
- Validation score for each fold
- Mean validation macro-F1
- Standard deviation of validation macro-F1
- Rank of the configuration
- Training score for each fold
- Mean training score

The most useful columns are:

```text
params
mean_test_score
std_test_score
rank_test_score
mean_train_score
```

`mean_test_score` is the mean cross-validation macro-F1. A lower
`rank_test_score` is better; rank 1 is the selected configuration.

The selected configuration and its final metrics are also copied into:

```text
outputs/model_results.csv
```

The columns `best_parameters`, `cv_macro_f1_mean`, and `cv_macro_f1_std`
summarize the sensitivity-analysis result.

## 9. Algorithm 1: K-Nearest Neighbors

### 9.1 How KNN works

KNN stores the training examples. To classify a new wine:

1. Calculate its distance from the training wines.
2. Select the nearest `k` wines.
3. Use their class labels to vote for the prediction.

Because distance is used, feature scaling is important.

### 9.2 KNN sensitivity parameters

The implementation tests:

```python
"classifier__n_neighbors": [3, 5, 7, 9, 11, 15, 21]
"classifier__weights": ["uniform", "distance"]
"classifier__metric": ["euclidean", "manhattan"]
```

There are:

```text
7 neighbor values × 2 weighting methods × 2 distance metrics = 28 configurations
```

### 9.3 Meaning of the KNN parameters

#### `n_neighbors`

This is the number of nearby training wines used for the vote.

- Small `k`: more sensitive to individual observations and possible noise.
- Large `k`: smoother predictions but may underfit.

#### `weights`

- `uniform`: every neighbor has equal importance.
- `distance`: closer neighbors receive more importance.

#### `metric`

- `euclidean`: straight-line distance.
- `manhattan`: sum of absolute feature differences.

### 9.4 KNN result

The best KNN configuration was:

```text
n_neighbors = 5
weights = distance
metric = euclidean
```

Its mean cross-validation macro-F1 was approximately:

```text
0.3364
```

Its final test results were:

```text
Accuracy:          0.6625
Balanced accuracy: 0.4087
Macro-F1:          0.4065
Weighted-F1:       0.6500
ROC-AUC:           0.7354
```

### 9.5 How to explain the KNN result

The best configuration uses a moderate neighborhood of five wines and gives
closer wines more influence. This indicates that local similarity is useful,
but using only one or two neighbors would probably be too sensitive to noise.

The accuracy is reasonably high because the model recognizes the dominant
quality 5 and 6 classes. However, its balanced accuracy and macro-F1 are much
lower because it performs poorly on rare classes such as quality 3 and 4.

## 10. Algorithm 2: Gaussian Naive Bayes

### 10.1 How Gaussian Naive Bayes works

Gaussian Naive Bayes calculates the probability of each quality class given
the feature values.

It assumes:

1. Each numerical feature follows a Gaussian distribution within a class.
2. Features are conditionally independent once the quality class is known.

The model then predicts the class with the highest posterior probability.

### 10.2 Gaussian Naive Bayes sensitivity parameter

The implementation tests:

```python
"classifier__var_smoothing": [
    1e-11,
    1e-10,
    1e-9,
    1e-8,
    1e-7,
    1e-6,
    1e-5,
]
```

Thus, seven smoothing values are tested.

### 10.3 Meaning of `var_smoothing`

`var_smoothing` adds a small stabilizing quantity to feature variances.

- Very small variance can make probability calculations unstable.
- Smoothing prevents numerical problems.
- Too much smoothing can oversimplify the distributions.

In this dataset, all tested values produced the same recorded cross-validation
score to the displayed precision:

```text
Mean CV macro-F1: approximately 0.3523
```

The first value, `1e-11`, was selected because it tied for rank 1.

### 10.4 Gaussian Naive Bayes result

The selected configuration was:

```text
var_smoothing = 1e-11
```

Its final test results were:

```text
Accuracy:          0.5625
Balanced accuracy: 0.3214
Macro-F1:          0.3204
Weighted-F1:       0.5681
ROC-AUC:           0.6838
```

### 10.5 How to explain the Gaussian Naive Bayes result

The model is very fast because its probability distributions can be estimated
directly from the training data. However, wine chemistry features are not
fully independent. For example, some acidity and sulfur-related measurements
are correlated.

Therefore, the Naive Bayes assumptions are restrictive. This helps explain why
the model is computationally efficient but has weaker classification
performance than KNN on this experiment.

## 11. Algorithm 3: Logistic Regression

### 11.1 How Logistic Regression works

Multinomial Logistic Regression estimates a probability for each quality
class. It combines the input features using a linear equation and converts
the resulting values into class probabilities.

It is an interpretable baseline because its coefficients show the direction
and strength of the relationship between a feature and a class.

### 11.2 Logistic Regression sensitivity parameters

The implementation tests:

```python
"classifier__C": [0.001, 0.01, 0.1, 1, 10, 100]
"classifier__class_weight": [None, "balanced"]
```

There are:

```text
6 regularization values × 2 class-weight options = 12 configurations
```

### 11.3 Meaning of the Logistic Regression parameters

#### `C`

`C` is the inverse of regularization strength.

- Small `C`: stronger regularization and a simpler model.
- Large `C`: weaker regularization and greater flexibility.

Testing multiple values checks whether the model is underfitting or
overfitting.

#### `class_weight`

- `None`: uses the original class frequencies.
- `balanced`: gives more weight to minority classes.

The balanced option can improve minority-class recall, but it can also reduce
performance on the majority classes.

### 11.4 Logistic Regression result

The best configuration was:

```text
C = 10
class_weight = None
```

Its mean cross-validation macro-F1 was approximately:

```text
0.3210
```

Its final test results were:

```text
Accuracy:          0.5844
Balanced accuracy: 0.2703
Macro-F1:          0.2762
Weighted-F1:       0.5639
ROC-AUC:           0.7519
```

### 11.5 How to explain the Logistic Regression result

The selected `C=10` indicates that, among the tested values, a less strongly
regularized model produced the best validation macro-F1. The unbalanced
option was selected because it performed better on the validation criterion
than the balanced option.

The model has the highest ROC-AUC of the three models. This means its
probability ranking is relatively good. However, its final hard class
predictions have lower macro-F1, especially for rare quality classes. ROC-AUC
and classification accuracy measure different aspects of performance, so it
is possible for one to be higher while the other is lower.

## 12. How are the final results calculated?

After the best parameters are selected using only the training data:

1. `GridSearchCV` refits the best pipeline on all training observations.
2. The model predicts the untouched test observations.
3. The predictions are compared with the true test labels.
4. The metrics are calculated from those predictions.

The implementation does this in `evaluate_model()`:

```python
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

The test set contains 320 wines, so for example:

```text
KNN accuracy = 0.6625
```

means:

```text
0.6625 × 320 = 212 correctly classified wines
```

The test metrics are not training metrics. They measure performance on data
that was held out during model selection.

## 13. Meaning of the reported metrics

### Accuracy

The fraction of all test wines classified correctly.

### Balanced accuracy

The average recall across all quality classes. It gives each class equal
importance.

### Precision

Of the wines predicted as a particular class, the fraction that truly belongs
to that class.

### Recall

Of the wines that truly belong to a particular class, the fraction correctly
identified.

### F1-score

The harmonic mean of precision and recall.

### Macro-F1

The F1-score is calculated separately for every class and averaged equally.
This is the primary tuning metric.

### Weighted-F1

The F1-score is averaged using class frequencies as weights. It reflects the
overall class distribution but can be dominated by quality 5 and 6.

### One-vs-rest ROC-AUC

Each class is temporarily treated as positive and the other classes as
negative. The resulting AUC values are averaged across classes.

### Log loss

This evaluates predicted probabilities. It penalizes a model heavily when it
is very confident but incorrect.

## 14. Why do accuracy and macro-F1 differ?

The test set contains many quality 5 and 6 wines but very few quality 3 and 8
wines. A model can obtain reasonable accuracy by predicting the dominant
classes while failing to identify rare classes.

For example, the KNN test report shows:

```text
Quality 3 F1-score: 0.0000
Quality 4 F1-score: 0.0000
Quality 5 F1-score: 0.7259
Quality 6 F1-score: 0.6312
Quality 7 F1-score: 0.6818
Quality 8 F1-score: 0.4000
```

The macro-F1 averages these class scores equally, so the zero scores for rare
classes reduce the macro-F1. This is why macro-F1 and balanced accuracy are
important in addition to accuracy.

## 15. Final comparison

The current run produced the following results:

| Model | Best parameters | CV macro-F1 | Test accuracy | Balanced accuracy | Test macro-F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| KNN | `k=5`, distance weights, Euclidean | 0.3364 | 0.6625 | 0.4087 | 0.4065 | 0.7354 |
| Gaussian Naive Bayes | `var_smoothing=1e-11` | 0.3523 | 0.5625 | 0.3214 | 0.3204 | 0.6838 |
| Logistic Regression | `C=10`, no class balancing | 0.3210 | 0.5844 | 0.2703 | 0.2762 | 0.7519 |

### Main conclusion

For this particular train-test split and macro-F1-based model selection:

- KNN has the highest test accuracy.
- KNN has the highest test macro-F1 and balanced accuracy.
- Gaussian Naive Bayes is the fastest model.
- Logistic Regression has the highest ROC-AUC.
- All models struggle with rare quality classes.

Therefore, KNN is the strongest model if the main objective is final class
prediction according to accuracy, balanced accuracy, and macro-F1. Logistic
Regression is still useful because it provides the best ROC-AUC and an
interpretable linear baseline.

These conclusions apply to this dataset, split, and parameter grid. They
should not be presented as universal claims about the algorithms.

## 16. How to reproduce the results

From the project directory, run:

```bash
.venv/bin/python wine_quality_models.py
```

The script will:

1. Download the dataset if necessary.
2. Generate the exploratory plots.
3. Create the stratified train-test split.
4. Run the three sensitivity analyses.
5. Select the best configuration for each algorithm.
6. Evaluate the selected models on the test set.
7. Write the CSV files and figures into `outputs/`.

## 17. Short answer for the instructor

> We used the UCI Red Wine Quality dataset with 1,599 observations, 11
> chemical features, and quality scores from 3 to 8. We first performed a
> stratified 80/20 train-test split. We kept the test set untouched and used
> five-fold stratified cross-validation on the training set. For sensitivity
> analysis, we varied KNN's number of neighbors, distance weighting, and
> distance metric; Gaussian Naive Bayes' variance smoothing; and Logistic
> Regression's regularization strength and class weighting. For every
> configuration, we calculated validation macro-F1. We selected the
> configuration with the highest mean validation macro-F1, refitted it on the
> complete training set, and evaluated it once on the test set. The different
> configurations and their scores are stored in the three sensitivity CSV
> files. The final metrics are stored in `model_results.csv`, and the
> class-level results are stored in `per_class_metrics.csv`. KNN gave the best
> test accuracy and macro-F1 in our run, while Logistic Regression gave the
> highest ROC-AUC.
