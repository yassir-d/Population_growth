import pandas as pd

def add_growth_rate(ts: pd.DataFrame) -> pd.DataFrame:
    """
    Add year-over-year population growth rate to the time series.
    """
    ts = ts.copy()
    ts["growth_rate"] = ts["population_total"].pct_change()
    return ts


def estimate_baseline_growth(
    ts: pd.DataFrame,
    start_year: int | None = None,
    end_year: int | None = None
) -> float:
    """
    Compute the average annual population growth rate over a given period.
    """
    ts = add_growth_rate(ts)

    # restrict period
    if start_year is not None:
        ts = ts[ts["year"] >= start_year]
    if end_year is not None:
        ts = ts[ts["year"] <= end_year]

    # drop NaN (first pct_change)
    avg_growth = ts["growth_rate"].dropna().mean()

    return avg_growth


def forecast_baseline(ts: pd.DataFrame, avg_growth: float, horizon: int) -> pd.DataFrame:
    """
    Forecast future population using a constant average annual growth rate.
    """
    ts = ts.copy()
    last_year = ts["year"].max()
    last_pop = ts["population_total"].iloc[-1]

    future_years = []
    future_population = []

    population = last_pop

    for h in range(1, horizon + 1):
        year = last_year + h
        population = population * (1 + avg_growth)  # apply growth
        future_years.append(year)
        future_population.append(population)

    df_future = pd.DataFrame({
        "year": future_years,
        "population_total": future_population
    })

    combined = pd.concat([ts, df_future], ignore_index=True)
    return combined
