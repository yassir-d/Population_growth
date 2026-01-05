from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def fit_ar_model(
        ml_df: pd.DataFrame,
        test_start_year: int = 2000,
        n_lags: int = 2,
) -> dict: 
    
    """
    Fit an autoregressive (AR) model using only lagged population values.
    """

    df = ml_df.copy()

    # AR features : only lagged population 
    lag_cols = [f"pop_lag_{i}" for i in range(1, n_lags + 1)]
    X = df[lag_cols]
    y = df["target_pop_next"]

    train = df["year"] < test_start_year
    test = df["year"] >= test_start_year

    X_train, X_test = X[train], X[test]
    y_train, y_test = y[train], y[test]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    rmse_train = np.sqrt(np.mean((y_train - y_train_pred) ** 2))
    rmse_test = np.sqrt(np.mean((y_test - y_test_pred) ** 2))

    print("AR model fitted.")
    print(f"AR order: {n_lags}")
    print(f"Train RMSE: {rmse_train:,.0f}")
    print(f"Test  RMSE: {rmse_test:,.0f}")
    print(f"Train n={len(X_train)}, Test n={len(X_test)}")

    results = {
        "model": f"AR({n_lags})",
        "train_rmse": float(rmse_train),
        "test_rmse": float(rmse_test),
        "n_lags": n_lags,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    return results





