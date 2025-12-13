import pandas as pd

def build_ml_table(ts : pd.DataFrame, n_lags: int = 1) -> pd.DataFrame: 
    """
    Turn the population time series into a supervised ML table.

    Parameters : 

    ts : pd.DataFrame
        Must contain : 
        - year 
        - population total 

    n_lags : int 
        How many lag features to create

        lag = 2 will create 2 lags :
        pop_lag_1 (pop at t-1)
        pop_lag_2 (pop at t-2)

    Returns

    pd.Dataframe
    Table with columns : 
        - year
        - population_toal
        - growth rate
        - pop_lag_1, pop_lag_2...
        - target_pop_next (population at t+1)
    """

    # 1) Making sure the data is sorted and copying it so not modifying the original
    ts = ts.sort_values("year").copy()

    # 2) Use float for population (better for ML)
    ts["population_total"] = ts["population_total"].astype(float)

    # 3) Year-over-year growth rate: (pop_t - pop_{t-1}) / pop_{t-1}
    ts["growth_rate"] = ts["population_total"].pct_change()

    # 4) Lag features: past population values
    # pop_lag_1 = previous year's population
    # pop_lag_2 = population 2 years ago, etc.

    for k in range(1, n_lags + 1):
        ts[f"pop_lag_{k}"] = ts["population_total"].shift(k)

    # 5) Target we want to predict : NExt year's population
    ts[f"target_pop_next"] = ts["population_total"].shift(-1)

    # 6) Drop rows with missing values (beginning + last row)
    ts_ml = ts.dropna().reset_index(drop=True)

    return ts_ml