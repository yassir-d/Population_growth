from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data_loader import load_population_timeseries
from src.baseline_model import estimate_baseline_growth, forecast_baseline

def main(test_start_year: int = 2000, start_year_for_growth: int = 1980, horizon: int = 20) -> None:
    # 1 Load yearly population data 
    ts = load_population_timeseries()

    # 2 Estimate baseline growth model from recent history (eg from 1980)
    avg_growth = estimate_baseline_growth(ts, start_year = 1980)
    print(f"Average annual growth since 1980: {avg_growth:.4%}")

    # Forecast 20 years into the future 
    combined = forecast_baseline(ts, avg_growth=avg_growth, horizon=20)

    last_historical_year = ts["year"].max()

    # Separate historical vs forecast rows 
    hist = combined[combined["year"] <= last_historical_year]
    fut = combined[combined["year"] > last_historical_year]

    # Plot: convert population to millions for nicer y-axis
    plt.figure(figsize=(10,5))
    plt.plot(
        hist["year"],
        hist["population_total"] / 1_000_000,
        label="Historical population",
    )
    plt.plot(
        fut["year"],
         fut["population_total"] / 1_000_000,
        "--",
        label="Baseline forecast",
    )

    plt.xlabel("Year")
    plt.ylabel("Population (millions)")
    plt.title("Swiss Population — Historical vs Baseline Constant-Growth Forecast")
    plt.legend()
    plt.tight_layout()

    # 5. Ensure results/ folder exists, then save figure
    results_dir = Path(__file__).resolve().parents[1] / "results" / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "baseline_forecast.png"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "baseline_forecast.png"
    plt.savefig(out_path, dpi=150)
    print(f" Saved plot to {out_path}")

    # Optional: show the window (if running locally with GUI)
    plt.close()


if __name__ == "__main__":
    main()

