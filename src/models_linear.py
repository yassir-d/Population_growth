from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def train_test_split_time(
    df: pd.DataFrame,
    test_start_year: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple time-based train/test split.

    All rows with year < test_start_year → train
    All rows with year >= test_start_year → test.
    """
    train = df[df["year"] < test_start_year].copy()
    test = df[df["year"] >= test_start_year].copy()
    return train, test


def fit_linear_model(df_ml: pd.DataFrame, test_start_year: int = 2000) -> None:
    """
    Fit a linear regression model to predict next year's population.

    Features used (if present in df_ml):
      - population_total
      - growth_rate
      - pop_lag_1
      - pop_lag_2

    The target is:
      - target_pop_next

    Prints RMSE on train and test.
    """
    # 1) Train/test split
    train, test = train_test_split_time(df_ml, test_start_year=test_start_year)

    if train.empty or test.empty:
        print("❌ Train or test set is empty. Check test_start_year.")
        print("Available years:", df_ml["year"].min(), "to", df_ml["year"].max())
        return

    # 2) Select feature columns that actually exist
    candidate_cols = ["population_total", "growth_rate", "pop_lag_1", "pop_lag_2"]
    feature_cols = [c for c in candidate_cols if c in df_ml.columns]

    if not feature_cols:
        print("❌ No feature columns found in df_ml. Check your feature engineering.")
        return

    # 3) Build X/y for train and test
    X_train = train[feature_cols].values
    y_train = train["target_pop_next"].values

    X_test = test[feature_cols].values
    y_test = test["target_pop_next"].values

    # 4) Fit linear regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 5) Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # 6) RMSE
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print("✅ Linear regression fitted.")
    print("Feature columns used:", feature_cols)
    print(f"Train RMSE: {rmse_train:,.0f}")
    print(f"Test  RMSE: {rmse_test:,.0f}")
    print(f"Train n={len(train)}, Test n={len(test)}")
