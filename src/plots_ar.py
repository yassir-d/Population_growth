from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from src.data_loader import load_population_timeseries
from src.features import build_ml_table


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main(test_start_year: int = 2000, n_lags: int = 2) -> None:
    # 1) Load & build ML table (includes lag features + target)
    ts = load_population_timeseries()
    df = build_ml_table(ts, n_lags=n_lags).copy()
    df = df.sort_values("year").reset_index(drop=True)

    # 2) Build AR(X) design matrix: predict next year's pop using ONLY lags
    lag_cols = [f"pop_lag_{k}" for k in range(1, n_lags + 1)]
    for c in lag_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column {c}. Available columns: {list(df.columns)}")

    X = df[lag_cols].values
    y = df["target_pop_next"].values
    years = df["year"].values  # year t (target is t+1)

    # 3) Time split
    train_mask = years < test_start_year
    test_mask = years >= test_start_year

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    years_train = years[train_mask]
    years_test = years[test_mask]

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Train or test is empty. Check test_start_year.")

    # 4) Fit AR model (LinearRegression on lag features)
    model = LinearRegression()
    model.fit(X_train, y_train)

    yhat_train = model.predict(X_train)
    yhat_test = model.predict(X_test)

    train_rmse = rmse(y_train, yhat_train)
    test_rmse = rmse(y_test, yhat_test)

    print("AR plot script ran.")
    print(f"AR({n_lags}) train RMSE: {train_rmse:,.0f}")
    print(f"AR({n_lags}) test  RMSE: {test_rmse:,.0f}")

    # 5) Output folder
    results_dir = Path(__file__).resolve().parents[1] / "results" / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Plot 1: Actual vs Pred (test period)
    plt.figure(figsize=(10, 5))
    # years_test is year t; target is pop at t+1, so label x as (t+1)
    plt.plot(years_test + 1, y_test, label="Actual (t+1)")
    plt.plot(years_test + 1, yhat_test, label="Predicted (t+1)")
    plt.title(f"AR({n_lags}) — Actual vs Predicted (Test from {test_start_year})")
    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.legend()
    out1 = results_dir / f"ar{n_lags}_actual_vs_pred_test.png"
    plt.tight_layout()
    plt.savefig(out1, dpi=200)
    plt.close()

    # --- Plot 2: Residuals (test)
    residuals = y_test - yhat_test
    plt.figure(figsize=(10, 5))
    plt.plot(years_test + 1, residuals)
    plt.axhline(0)
    plt.title(f"AR({n_lags}) — Residuals (Actual - Predicted), Test")
    plt.xlabel("Year")
    plt.ylabel("Residual")
    out2 = results_dir / f"ar{n_lags}_residuals_test.png"
    plt.tight_layout()
    plt.savefig(out2, dpi=200)
    plt.close()

    # --- Plot 3: Residual histogram (test)
    plt.figure(figsize=(10, 5))
    plt.hist(residuals, bins=20)
    plt.title(f"AR({n_lags}) — Residual Distribution (Test)")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    out3 = results_dir / f"ar{n_lags}_residual_hist_test.png"
    plt.tight_layout()
    plt.savefig(out3, dpi=200)
    plt.close()

    print(f"Saved:\n- {out1}\n- {out2}\n- {out3}")


if __name__ == "__main__":
    main(test_start_year=2000, n_lags=2)


"""
AR model diagnostics and forecast plots.

This script refits an AR(p) model on lagged population data
and produces diagnostic plots on the test period.
"""
