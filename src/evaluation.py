from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def evaluate_baseline_constant_growth(    
    ts: pd.DataFrame,
    test_start_year: int = 2000,
    start_year_for_growth: int = 1980,
) -> dict: 
    """
    Evaluate a constant growth constant model.

    - Estimate average growth from start_year_for_growth up to just before test_start_year
    - Forecast forward in the test period
    - Compute RMSE for the test period
    """

    ts = ts.sort_values("year").copy()
    ts["population_total"] = ts["population_total"].astype(float)

    train = ts[ts["year"] < test_start_year].copy()
    test = ts[ts["year"] >= test_start_year].copy()

    # Use only part of the training window to estimate growth (1980)
    growth_window = train[train["year"] >= start_year_for_growth].copy()

    # Compute average annual growth rate from the window 
    growth_window["growth_rate"] = growth_window["population_total"].pct_change()
    avg_growth = growth_window ["growth_rate"].dropna().mean()

    # Forecast for each year in test period, recursively
    last_pop = train["population_total"].iloc[-1]
    preds = []
    for _ in range (len(test)): 
        last_pop = last_pop * (1 + avg_growth)
        preds.append(last_pop)

    y_true = test["population_total"].values
    y_pred = np.array(preds)

    test_rmse = rmse(y_true, y_pred)
    train_rmse = float("nan")

    print("Baseline constant growth evaluated.")
    print(f"Estimated avg growth (from {start_year_for_growth}): {avg_growth*100:.4f}%")
    print(f"Test RMSE: {test_rmse:,.0f}")
    print(f"Train n={len(train)}, Test n={len(test)}")

    results = {
        "model": "Baseline constant growth",
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "avg_growth": avg_growth,
        "train_size": len(train),
        "test_size": len(test),
    }

    return results
