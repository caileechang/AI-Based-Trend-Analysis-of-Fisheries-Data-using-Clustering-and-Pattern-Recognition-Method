import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing import prepare_yearly, prepare_monthly


file_path = "tests/Testing data.xlsx"

def test_integration_real_data_yearly():
    df_land = pd.read_excel(file_path, sheet_name="Fish Landing")
    df_vess = pd.read_excel(file_path, sheet_name="Fish Vessels")

    result = prepare_yearly(df_land, df_vess)

    assert not result.empty
    assert "Total Fish Landing (Tonnes)" in result.columns
    assert "Total number of fishing vessels" in result.columns
    assert result["Year"].nunique() >= 1
    assert result["Total Fish Landing (Tonnes)"].sum() > 0


def test_integration_real_data_monthly():
    df_land = pd.read_excel(file_path, sheet_name="Fish Landing")

    result = prepare_monthly(df_land, None)

    assert not result.empty
    assert result["Month"].between(1, 12).all()
    assert result["Fish Landing (Tonnes)"].sum() > 0
