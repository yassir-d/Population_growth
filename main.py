from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_results_folders() -> dict[str, Path]:
    root = Path(__file__).resolve().parent
    results = root / "results"
    figures = results / "figures"
    tables = results / "tables"

    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    return {"root": root, "results": results, "figures": figures, "tables": tables}


def run_model_comparison(test_start_year: int = 2000) -> pd.DataFrame:
    # Comparison pipeline
    from src.compare_models import compare_all_models

    df = compare_all_models(test_start_year=test_start_year)
    return df


def save_comparison_table(df: pd.DataFrame, out_path: Path) -> None:
    df.to_csv(out_path, index=False)
    print(f" Saved model comparison table → {out_path}")


def run_plots(test_start_year: int = 2000, n_lags: int = 2) -> None:
    """
    Tries to run plot scripts if they exist.
    If a plot module is missing, it will just skip it (no crash).
    """
    # Baseline plot
    try:
        from src.plots_baseline import main as baseline_plot_main

        baseline_plot_main(test_start_year=test_start_year)
        print("Baseline plots generated.")
    except Exception as e:
        print(f"⚠️ Skipped baseline plots (src.plots_baseline). Reason: {e}")

    # Linear regression plots
    try:
        from src.plots_linear import main as linear_plot_main

        linear_plot_main(test_start_year=test_start_year, n_lags=n_lags)
        print("Linear regression plots generated.")
    except Exception as e:
        print(f"⚠️ Skipped linear plots (src.plots_linear). Reason: {e}")

    # AR plots
    try:
        from src.plots_ar import main as ar_plot_main

        ar_plot_main(test_start_year=test_start_year, n_lags=n_lags)
        print("AR plots generated.")
    except Exception as e:
        print(f"⚠️ Skipped AR plots (src.plots_ar). Reason: {e}")


def main() -> None:
    paths = ensure_results_folders()

    print("\n=== Population Growth Project: Running main pipeline ===\n")

    # 1) Compare models
    df = run_model_comparison(test_start_year=2000)
    print("\nModel comparison table:\n")
    print(df)

    # 2) Save table
    out_csv = paths["tables"] / "model_comparison.csv"
    save_comparison_table(df, out_csv)

    # 3) Plots (optional but recommended)
    run_plots(test_start_year=2000, n_lags=2)

    print("\n=== Done. Check results/figures and results/tables ===\n")


if __name__ == "__main__":
    main()
