import pandas as pd
import sys, os
import pytest

# Make project root importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from data_processing import prepare_yearly, prepare_monthly

# Absolute path to Testing data.xlsx
FILE_PATH = os.path.join(os.path.dirname(__file__), "Testing data.xlsx")


def test_system_end_to_end_real_data():
    """
    System Test:
    End-to-end validation using real fisheries dataset.
    This test simulates how the system behaves when a user uploads data.
    """

    # ===== STEP 1: Load real dataset =====
    df_land = pd.read_excel(FILE_PATH, sheet_name="Fish Landing")
    df_vess = pd.read_excel(FILE_PATH, sheet_name="Fish Vessels")

    assert not df_land.empty
    assert not df_vess.empty

    # ===== STEP 2: Run yearly processing =====
    yearly_result = prepare_yearly(df_land, df_vess)

    assert not yearly_result.empty
    assert "Total Fish Landing (Tonnes)" in yearly_result.columns
    assert "Total number of fishing vessels" in yearly_result.columns

    # Logical system checks
    assert yearly_result["Total Fish Landing (Tonnes)"].sum() > 0
    assert yearly_result["Total number of fishing vessels"].sum() >= 0

    # ===== STEP 3: Run monthly processing =====
    monthly_result = prepare_monthly(df_land)

    assert not monthly_result.empty
    assert monthly_result["Month"].between(1, 12).all()
    assert monthly_result["Fish Landing (Tonnes)"].sum() > 0

    # ===== STEP 4: Final system-level validation =====
    assert yearly_result["Year"].nunique() >= 1
    assert monthly_result["Year"].nunique() >= 1
