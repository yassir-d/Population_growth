from pathlib import Path
import pandas as pd
from pyaxis import pyaxis

# Find the project root 
BASE_DIR = Path(__file__).resolve().parents[1]

# Path to data/raw/ for POPULATION_RESIDENT_DEMOG.XLSX
RAW_DATA_DIR = BASE_DIR / "data" / "raw"


def load_population_raw() -> pd.DataFrame:
    """
    Load the raw permanent resident population Excel file.
    """
    path = RAW_DATA_DIR / "permanent_resident_demog.xlsx"
    df = pd.read_excel(path)
    return df


# For POP_SEX_AGE.CSV
def load_pop_sex_age_raw() -> pd.DataFrame:
    """
    Load the BFS .px file for population by sex and age.
    Returns a pandas DataFrame with the actual DATA table.
    """
    path = RAW_DATA_DIR / "Pop_sex_age.px"

    tables = pyaxis.parse(str(path), encoding="latin-1")

    print("Multilingual PX file")
    print("Type returned by pyaxis:", type(tables))
    print("Top-level keys:", tables.keys())

    # 1) Get the DATA part (numbers)
    data_part = tables.get("DATA", None)
    if data_part is None:
        raise ValueError("DATA key not found in PX structure")

    print("Type of DATA part:", type(data_part))

    # 2) If DATA is already a DataFrame, we are done
    if isinstance(data_part, pd.DataFrame):
        df = data_part
    # 3) If DATA is a dict (e.g. different languages or tables), take the first table
    elif isinstance(data_part, dict):
        print("DATA subkeys:", data_part.keys())
        # Take the first DataFrame among the values
        first_value = list(data_part.values())[0]
        if isinstance(first_value, pd.DataFrame):
            df = first_value
        else:
            raise TypeError("DATA substructure is not a DataFrame; got type: "
                            f"{type(first_value)}")
    else:
        raise TypeError(f"Unexpected DATA type: {type(data_part)}")

    print("Loaded DATA table with shape:", df.shape)
    print("Columns:", df.columns)

    return df

# Time series
def load_population_timeseries() -> pd.DataFrame:
    """
    Return a simple yearly total population time series for Switzerland.

    Output columns:
    - year: int
    - population_total: float
    """
    # 1. Load the raw BFS PX data as a DataFrame
    df = load_pop_sex_age_raw()

    # 2. Keep only rows where sex is 'total' AND age is 'total'
    mask_total = (
        (df["Geschlecht"] == "Geschlecht - Total") &
        (df["Alter"] == "Alter - Total")
    )
    df_total = df.loc[mask_total].copy()

    # 3. Clean up types and sort
    df_total["Jahr"] = df_total["Jahr"].astype(int)
    df_total = df_total.sort_values("Jahr")

    # 4. Rename columns jahr to year
    df_total = df_total.rename(columns={
        "Jahr": "year",
        "DATA": "population_total",
    })

    # 5. Keep only the columns needed
    df_total = df_total[["year", "population_total"]].reset_index(drop=True)
    df_total["population_total"] = df_total["population_total"].astype(int)  

    print("Yearly population time series shape:", df_total.shape)
    return df_total



