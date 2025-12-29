import pandas as pd
import pytest
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing import prepare_yearly

def sample_data():
    df_land = pd.DataFrame({
        "Year": [2024, 2024],
        "State": ["Johor", "Selangor"],
        "Fish Landing (Tonnes)": [100, 150]
    })

    df_vess = pd.DataFrame({
        "Year": [2024, 2024],
        "State": ["Johor", "Selangor"],
        "Inboard Powered": [10, 20],
        "Outboard Powered": [5, 10],
        "Non-Powered": [2, 3]
    })

    return df_land, df_vess


# Basic logical checks
def test_prepare_yearly_basic():
    df_land, df_vess = sample_data()
    result = prepare_yearly(df_land, df_vess)

    assert not result.empty
    assert "Total Fish Landing (Tonnes)" in result.columns
    assert "Total number of fishing vessels" in result.columns
    assert result["Total Fish Landing (Tonnes)"].sum() == 250


# Vessel sum correctness
def test_vessel_sum():
    df_land, df_vess = sample_data()
    result = prepare_yearly(df_land, df_vess)

    johor_vessels = result[result["State"].str.upper() == "JOHOR"]["Total number of fishing vessels"].iloc[0]

    assert johor_vessels == 10 + 5 + 2


#  Missing values filled as 0
def test_missing_vessel_values():
    df_land = pd.DataFrame({
        "Year": [2024],
        "State": ["Johor"],
        "Fish Landing (Tonnes)": [300]
    })
    df_vess = pd.DataFrame({
        "Year": [2024],
        "State": ["Johor"],
        "Inboard Powered": [None],
        "Outboard Powered": [10],
        "Non-Powered": [None]
    })
    result = prepare_yearly(df_land, df_vess)
    assert result["Total number of fishing vessels"].iloc[0] == 10  # NaN -> 0
# Duplicate State-Year rows aggregated
def test_duplicate_rows():
    df_land = pd.DataFrame({
        "Year": [2024, 2024],
        "State": ["Johor", "Johor"],
        "Fish Landing (Tonnes)": [100, 200]
    })
    df_vess = pd.DataFrame({
        "Year": [2024],
        "State": ["Johor"],
        "Inboard Powered": [10],
        "Outboard Powered": [5],
        "Non-Powered": [2]
    })
    result = prepare_yearly(df_land, df_vess)
    assert result["Total Fish Landing (Tonnes)"].iloc[0] == 300
#  Missing required columns → must raise error
def test_missing_columns_error():
    df_land, df_vess = sample_data()
    df_land = df_land.drop(columns=["Fish Landing (Tonnes)"])  # corrupt input
    with pytest.raises(Exception):
        prepare_yearly(df_land, df_vess)
# Different years both included
def test_multiple_years():
    df_land, df_vess = sample_data()
    df_land.loc[len(df_land)] = [2025, "Johor", 500]
    df_vess.loc[len(df_vess)] = [2025, "Johor", 20, 10, 5]
    result = prepare_yearly(df_land, df_vess)
    assert result["Year"].nunique() == 2

