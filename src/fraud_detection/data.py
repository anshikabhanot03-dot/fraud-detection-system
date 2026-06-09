from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from fraud_detection.config import RANDOM_STATE, TARGET_COLUMN, TEST_SIZE


def load_creditcard_data(path) -> pd.DataFrame:
    """Load the Kaggle European credit card fraud dataset."""
    data = pd.read_csv(path)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in {path}")
    return data


def split_features_target(data: pd.DataFrame):
    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN].astype(int)
    return X, y


def stratified_train_test_split(X, y, test_size: float = TEST_SIZE):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )

