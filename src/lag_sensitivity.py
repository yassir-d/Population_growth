from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import load_population_timeseries
from src.features import build_ml_table
from src.models_ar import fit_ar_model


def ensure_results_dirs() -> tuple[Path, Path]:
    """Create results folders if they don't exist."""
    base = Path(__file__).resolve().parents[1]  # project root
    figures = base / "results" / "figures"
    tables = base / "results" / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    return figures, tables


def run_lag_sensitivity(
    test_start_year: int = 2000,
    max_lag: int = 5,
) -> pd.DataFrame:
    """
    Fit AR(p) models for p=1..max_lag and return a comparison DataFrame.
    Uses the same ML table builder so columns match pop_lag_1, pop_lag_2, ...
    """
    ts = load_population_timeseries()
    results: list[dict] = []

    for p in range(1, max_lag + 1):
        ml = build_ml_table(ts, n_lags=p)
        res = fit_ar_model(ml, test_start_year=test_start_year, n_lags=p)
        results.append(res)

    df = pd.DataFrame(results)
    return df


def plot_rmse_vs_lag(df: pd.DataFrame, out_path: Path) -> None:
    """Save a plot of test RMSE vs AR lag length."""
    df_plot = df.copy()
    df_plot = df_plot.sort_values("n_lags")

    plt.figure(figsize=(9, 5))
    plt.plot(df_plot["n_lags"], df_plot["test_rmse"], marker="o")
    plt.xlabel("AR lag order (p)")
    plt.ylabel("Test RMSE")
    plt.title("Lag sensitivity: AR(p) test error")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    figures_dir, tables_dir = ensure_results_dirs()

    df = run_lag_sensitivity(test_start_year=2000, max_lag=5)

    csv_path = tables_dir / "lag_sensitivity_ar.csv"
    fig_path = figures_dir / "lag_sensitivity_ar.png"

    df.to_csv(csv_path, index=False)
    plot_rmse_vs_lag(df, fig_path)

    print("\n✅ Lag sensitivity finished.")
    print(f"Saved table:  {csv_path}")
    print(f"Saved figure: {fig_path}")
    print("\nResults preview:\n")
    print(df[["model", "n_lags", "train_rmse", "test_rmse", "train_size", "test_size"]])


if __name__ == "__main__":
    main()
