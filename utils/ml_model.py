from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


MINIMUM_RECORDS = 30
MINIMUM_CLASS_COUNT = 8
RANDOM_SEED = 42


def _unavailable(reason: str) -> dict[str, Any]:
    return {"available": False, "reason": reason}


def train_promotion_model(df: pd.DataFrame) -> dict[str, Any]:
    """Train and evaluate only when both target classes support a holdout test."""
    if "promotion_status" not in df.columns:
        return _unavailable("A promotion_status target column is required; no model was trained.")
    valid = df[df["promotion_status"].isin(["Yes", "No"])].copy()
    if len(valid) < MINIMUM_RECORDS:
        return _unavailable(
            f"Only {len(valid)} records have a recognized Yes/No target. At least {MINIMUM_RECORDS} are required for this demonstration."
        )
    counts = valid["promotion_status"].value_counts()
    if len(counts) != 2:
        return _unavailable("The target must contain both Yes and No outcomes; no model was trained.")
    if int(counts.min()) < MINIMUM_CLASS_COUNT:
        return _unavailable(
            f"The smallest target group has {int(counts.min())} records. At least {MINIMUM_CLASS_COUNT} per outcome is required for a stratified evaluation."
        )

    feature_columns = [
        column
        for column in valid.columns
        if column not in {"promotion_status", "gender"}
        and valid[column].notna().any()
    ]
    if not feature_columns:
        return _unavailable("No eligible non-gender input features are available; no model was trained.")

    features = valid[feature_columns].copy()
    target = valid["promotion_status"]
    numeric_columns = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_columns:
        numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=RANDOM_SEED,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    test_count = max(4, int(np.ceil(len(valid) * 0.25)))
    if len(valid) - test_count < 4 or test_count >= len(valid):
        return _unavailable("There are not enough records to keep suitable training and test groups.")
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=test_count,
            random_state=RANDOM_SEED,
            stratify=target,
        )
        if y_test.value_counts().min() < 2:
            return _unavailable("The holdout set would contain too few examples of an outcome for a useful evaluation.")
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        labels = ["No", "Yes"]
        matrix = confusion_matrix(y_test, predictions, labels=labels)
        importances = pipeline.named_steps["classifier"].feature_importances_
        names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        feature_importance = pd.DataFrame(
            {"feature": names, "importance": importances}
        ).sort_values("importance", ascending=False, ignore_index=True)
        feature_specs: dict[str, dict[str, Any]] = {}
        defaults: dict[str, Any] = {}
        for column in feature_columns:
            if column in numeric_columns:
                feature_specs[column] = {"kind": "numeric"}
                defaults[column] = float(x_train[column].median())
            else:
                options = sorted(x_train[column].dropna().astype(str).unique().tolist())
                if not options:
                    continue
                feature_specs[column] = {"kind": "categorical", "options": options}
                defaults[column] = str(x_train[column].mode().iloc[0])
    except (ValueError, TypeError, AttributeError) as error:
        return _unavailable(f"Model training or holdout evaluation could not be completed: {error}")

    return {
        "available": True,
        "pipeline": pipeline,
        "feature_columns": feature_columns,
        "feature_specs": feature_specs,
        "defaults": defaults,
        "feature_importance": feature_importance,
        "labels": labels,
        "confusion_matrix": matrix.tolist(),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, pos_label="Yes", zero_division=0)),
        "recall": float(recall_score(y_test, predictions, pos_label="Yes", zero_division=0)),
        "f1": float(f1_score(y_test, predictions, pos_label="Yes", zero_division=0)),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "class_counts": {str(key): int(value) for key, value in counts.items()},
        "gender_excluded": True,
        "random_seed": RANDOM_SEED,
    }
