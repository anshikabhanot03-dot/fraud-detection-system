from __future__ import annotations

import argparse

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from fraud_detection.config import (
    FALSE_NEGATIVE_COST,
    FALSE_POSITIVE_COST,
    METRICS_PATH,
    MODEL_DIR,
    MODEL_PATH,
    RANDOM_STATE,
    RAW_DATA_PATH,
    THRESHOLD_PATH,
)
from fraud_detection.data import (
    load_creditcard_data,
    split_features_target,
    stratified_train_test_split,
)
from fraud_detection.evaluation import (
    classification_metrics,
    save_json,
    select_min_cost_threshold,
    threshold_cost_curve,
)


def build_models(y_train):
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    scale_pos_weight = negative_count / max(positive_count, 1)

    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "xgboost": XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def isolation_forest_scores(X_train, X_test):
    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train)
    raw_scores = -model.decision_function(X_test)
    min_score = raw_scores.min()
    max_score = raw_scores.max()
    if np.isclose(max_score, min_score):
        return np.zeros_like(raw_scores)
    return (raw_scores - min_score) / (max_score - min_score)


def train(data_path=RAW_DATA_PATH):
    data = load_creditcard_data(data_path)
    X, y = split_features_target(data)
    X_train, X_test, y_train, y_test = stratified_train_test_split(X, y)

    models = build_models(y_train)
    results = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        cost_curve = threshold_cost_curve(
            y_test,
            y_proba,
            false_positive_cost=FALSE_POSITIVE_COST,
            false_negative_cost=FALSE_NEGATIVE_COST,
        )
        best_threshold = select_min_cost_threshold(cost_curve)["threshold"]
        results[name] = {
            "metrics_at_min_cost_threshold": classification_metrics(
                y_test,
                y_proba,
                threshold=best_threshold,
            ),
            "min_cost_threshold": select_min_cost_threshold(cost_curve),
        }
        trained_models[name] = model

    iso_scores = isolation_forest_scores(X_train, X_test)
    iso_cost_curve = threshold_cost_curve(
        y_test,
        iso_scores,
        false_positive_cost=FALSE_POSITIVE_COST,
        false_negative_cost=FALSE_NEGATIVE_COST,
    )
    iso_threshold = select_min_cost_threshold(iso_cost_curve)["threshold"]
    results["isolation_forest"] = {
        "metrics_at_min_cost_threshold": classification_metrics(
            y_test,
            iso_scores,
            threshold=iso_threshold,
        ),
        "min_cost_threshold": select_min_cost_threshold(iso_cost_curve),
    }

    selected_model_name = "xgboost"
    selected_threshold = results[selected_model_name]["min_cost_threshold"]["threshold"]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(trained_models[selected_model_name], MODEL_PATH)
    joblib.dump(float(selected_threshold), THRESHOLD_PATH)
    save_json(
        {
            "selected_model": selected_model_name,
            "selected_threshold": selected_threshold,
            "false_positive_cost": FALSE_POSITIVE_COST,
            "false_negative_cost": FALSE_NEGATIVE_COST,
            "model_results": results,
            "features": list(X.columns),
        },
        METRICS_PATH,
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="Train fraud detection models.")
    parser.add_argument(
        "--data-path",
        default=RAW_DATA_PATH,
        help="Path to creditcard.csv",
    )
    args = parser.parse_args()
    train(args.data_path)


if __name__ == "__main__":
    main()

