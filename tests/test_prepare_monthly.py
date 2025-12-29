import pandas as pd
import pytest
from data_processing import prepare_monthly

#  Basic correctness test
def test_prepare_monthly_basic():
    df_land = pd.DataFrame({
        "Year": [2024],
        "Month": [3],  # Already numeric
        "State": ["Johor"],
        "Fish Landing (Tonnes)": [150]
    })

    result = prepare_monthly(df_land)

    assert not result.empty
    assert result["Year"].iloc[0] == 2024
    assert result["Month"].iloc[0] == 3
    assert result["Fish Landing (Tonnes)"].iloc[0] == 150


#  Ensure month names convert to numbers
def test_month_name_conversion():
    df_land = pd.DataFrame({
        "Year": [2024],
        "Month": ["April"],
        "State": ["Johor"],
        "Fish Landing (Tonnes)": [100]
    })

    result = prepare_monthly(df_land)

    assert result["Month"].iloc[0] == 4


#  State standardization test
def test_state_standardization():
    df_land = pd.DataFrame({
        "Year": [2024],
        "Month": [2],
        "State": ["JOHOR TIMUR"],
        "Fish Landing (Tonnes)": [200]
    })

    result = prepare_monthly(df_land)

    assert result["State"].iloc[0] == "JOHOR"  # standardized


#  Duplicate rows aggregated correctly
def test_duplicate_month_state():
    df_land = pd.DataFrame({
        "Year": [2024, 2024],
        "Month": [1, 1],
        "State": ["Johor", "Johor"],
        "Fish Landing (Tonnes)": [50, 75]
    })

    result = prepare_monthly(df_land)
    assert result["Fish Landing (Tonnes)"].iloc[0] == 125


#  Missing required column should raise error
def test_missing_column_error():
    df_land = pd.DataFrame({
        "Year": [2024],
        "State": ["Johor"],
        "Fish Landing (Tonnes)": [150]
    })

    with pytest.raises(Exception):
        prepare_monthly(df_land)
