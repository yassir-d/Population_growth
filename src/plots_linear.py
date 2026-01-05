import matplotlib.pyplot as plt
from pathlib import Path

from src.data_loader import load_population_timeseries
from src.features import build_ml_table
from src.models_linear import train_test_split_time
from sklearn.linear_model import LinearRegression

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main(test_start_year: int = 2000, n_lags: int = 2) -> None:
    ts = load_population_timeseries()
    ml = build_ml_table(ts, n_lags=2)

    train, test = train_test_split_time(ml, test_start_year)

    feature_cols = ["population_total", "growth_rate", "pop_lag_1", "pop_lag_2"]

    X_train = train[feature_cols]
    y_train = train["target_pop_next"]
    X_test = test[feature_cols]
    y_test = test["target_pop_next"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # ---- Actual vs Predicted ----
    plt.figure(figsize=(10, 5))
    plt.plot(test["year"], y_test, label="Actual")
    plt.plot(test["year"], y_pred, label="Predicted")
    plt.title("Linear Regression: Actual vs Predicted Population")
    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "linear_actual_vs_pred.png")
    plt.close()

    # ---- Residuals ----
    residuals = y_test - y_pred

    plt.figure(figsize=(10, 5))
    plt.plot(test["year"], residuals)
    plt.axhline(0, linestyle="--", color="black")
    plt.title("Linear Regression: Residuals")
    plt.xlabel("Year")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "linear_residuals.png")
    plt.close()


if __name__ == "__main__":
    main()