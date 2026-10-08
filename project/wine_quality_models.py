"""Compare KNN, Gaussian Naive Bayes, and Logistic Regression.

The experiment uses the UCI red wine quality dataset, stratified 80/20
train-test splitting, and 5-fold cross-validation for hyperparameter tuning.
The test set is used once, after model selection, for final evaluation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize

RANDOM_STATE = 42
DATA_URL = "https://archive.ics.uci.edu/static/public/186/wine+quality.zip"


def load_data(data_path: Path | None, data_dir: Path) -> pd.DataFrame:
    """Load a local red-wine CSV or download the UCI archive if absent."""
    if data_path is not None:
        return pd.read_csv(data_path, sep=";")

    archive_path = data_dir / "wine_quality.zip"
    csv_path = data_dir / "winequality-red.csv"
    data_dir.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        with urlopen(DATA_URL, timeout=60) as response:
            archive_path.write_bytes(response.read())
        with ZipFile(archive_path) as archive:
            archive.extract("winequality-red.csv", data_dir)
    return pd.read_csv(csv_path, sep=";")


def build_models() -> dict[str, tuple[Pipeline, dict[str, list[object]]]]:
    """Create leakage-safe pipelines and their sensitivity-analysis grids."""
    scaled = StandardScaler()
    return {
        "KNN": (
            Pipeline(
                [("scaler", scaled), ("classifier", KNeighborsClassifier())]
            ),
            {
                "classifier__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
                "classifier__weights": ["uniform", "distance"],
                "classifier__metric": ["euclidean", "manhattan"],
            },
        ),
        "Gaussian Naive Bayes": (
            Pipeline(
                [("scaler", scaled), ("classifier", GaussianNB())]
            ),
            {
                "classifier__var_smoothing": [
                    1e-11,
                    1e-10,
                    1e-9,
                    1e-8,
                    1e-7,
                    1e-6,
                    1e-5,
                ]
            },
        ),
        "Logistic Regression": (
            Pipeline(
                [
                    ("scaler", scaled),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=2000,
                            solver="lbfgs",
                        ),
                    ),
                ]
            ),
            {
                "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
    }


def save_eda_plots(data: pd.DataFrame, output_dir: Path) -> None:
    """Save target distribution and feature-correlation plots."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=data, x="quality", ax=ax, color="#4472C4")
    ax.set(title="Red Wine Quality Class Distribution", xlabel="Quality", ylabel="Count")
    fig.tight_layout()
    fig.savefig(str(output_dir / "quality_distribution.png"), dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data.corr(numeric_only=True), cmap="vlag", center=0, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(str(output_dir / "correlation_heatmap.png"), dpi=160)
    plt.close(fig)


def save_confusion_matrix(
    y_true: pd.Series, predictions: pd.Series, model_name: str, output_dir: Path
) -> None:
    labels = sorted(y_true.unique())
    matrix = confusion_matrix(y_true, predictions, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set(
        title=f"{model_name} Confusion Matrix",
        xlabel="Predicted quality",
        ylabel="True quality",
    )
    fig.tight_layout()
    filename = model_name.lower().replace(" ", "_") + "_confusion_matrix.png"
    fig.savefig(str(output_dir / filename), dpi=160)
    plt.close(fig)


def evaluate_model(
    model: GridSearchCV,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    output_dir: Path,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Evaluate a selected model and return aggregate and per-class metrics."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    classes = model.classes_
    y_test_binary = label_binarize(y_test, classes=classes)
    aggregate = {
        "model": model_name,
        "best_parameters": model.best_params_,
        "cv_macro_f1_mean": model.best_score_,
        "cv_macro_f1_std": model.cv_results_["std_test_score"][model.best_index_],
        "test_accuracy": accuracy_score(y_test, predictions),
        "test_balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "test_macro_precision": precision_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "test_macro_recall": recall_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "test_macro_f1": f1_score(y_test, predictions, average="macro"),
        "test_weighted_f1": f1_score(y_test, predictions, average="weighted"),
        "test_roc_auc_ovr_macro": roc_auc_score(
            y_test_binary, probabilities, multi_class="ovr", average="macro"
        ),
        "test_log_loss": log_loss(y_test, probabilities, labels=classes),
    }
    report = pd.DataFrame(
        classification_report(
            y_test, predictions, labels=classes, output_dict=True, zero_division=0
        )
    ).T.reset_index(names="quality")
    save_confusion_matrix(y_test, pd.Series(predictions), model_name, output_dir)
    return aggregate, report


def save_comparison_plot(results: pd.DataFrame, output_dir: Path) -> None:
    metric_columns = [
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
    ]
    chart_data = results.set_index("model")[metric_columns]
    ax = chart_data.plot(kind="bar", figsize=(11, 6), ylim=(0, 1))
    ax.set(title="Model Performance Comparison", xlabel="", ylabel="Score")
    ax.legend(loc="lower right")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(str(output_dir / "model_comparison.png"), dpi=160)
    plt.close()


def run_experiment(data_path: Path | None, output_dir: Path, data_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(data_path, data_dir)
    if "quality" not in data.columns:
        raise ValueError("Expected a 'quality' target column in the input CSV.")
    if data.isnull().any().any():
        raise ValueError("Input data contains missing values; clean it before training.")

    save_eda_plots(data, output_dir)
    X = data.drop(columns="quality")
    y = data["quality"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results: list[dict[str, object]] = []
    reports: list[pd.DataFrame] = []
    for model_name, (pipeline, parameter_grid) in build_models().items():
        started = time.perf_counter()
        search = GridSearchCV(
            pipeline,
            parameter_grid,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            return_train_score=True,
        )
        search.fit(X_train, y_train)
        result, report = evaluate_model(search, X_test, y_test, model_name, output_dir)
        result["fit_time_seconds"] = time.perf_counter() - started
        results.append(result)
        report.insert(0, "model", model_name)
        reports.append(report)

        sensitivity = pd.DataFrame(search.cv_results_)
        sensitivity.to_csv(
            output_dir / f"{model_name.lower().replace(' ', '_')}_sensitivity.csv",
            index=False,
        )

    results_frame = pd.DataFrame(results)
    results_frame["best_parameters"] = results_frame["best_parameters"].map(json.dumps)
    results_frame.to_csv(output_dir / "model_results.csv", index=False)
    pd.concat(reports, ignore_index=True).to_csv(
        output_dir / "per_class_metrics.csv", index=False
    )
    save_comparison_plot(results_frame, output_dir)
    print(results_frame.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Optional local semicolon-delimited CSV.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs"), help="Directory for metrics and plots."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"), help="Directory for downloaded data."
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_experiment(arguments.data, arguments.output_dir, arguments.data_dir)
