from pathlib import Path

import pandas as pd

from src.data_loader import load_population_timeseries
from src.features import build_ml_table
from src.evaluation import evaluate_baseline_constant_growth
from src.models_linear import fit_linear_model
from src.models_ar import fit_ar_model


def compare_all_models(test_start_year=2000):
    ts = load_population_timeseries()
    ml = build_ml_table(ts, n_lags=2)

    results = []

    # Baseline
    baseline_res = evaluate_baseline_constant_growth(
        ts,
        test_start_year=test_start_year,
        start_year_for_growth=1980,
    )
    results.append(baseline_res)

    # Linear regression
    linear_res = fit_linear_model(
        ml,
        test_start_year=test_start_year,
    )
    results.append(linear_res)

    # AR model
    ar_res = fit_ar_model(
        ml,
        test_start_year=test_start_year,
        n_lags=2,
    )
    results.append(ar_res)

    return pd.DataFrame(results)


if __name__ == "__main__":
    df = compare_all_models()
    print("\n Model comparison table:\n")
    print(df)

    RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "tables"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    out_path = RESULTS_DIR / "model_comparison.csv"
    df.to_csv(out_path, index=False)
    print(f"\n Saved comparison table to {out_path}")
