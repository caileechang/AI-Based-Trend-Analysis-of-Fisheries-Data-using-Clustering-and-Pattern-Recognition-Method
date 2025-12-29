import pandas as pd

# --- Helper: Convert Month Name to Number (if needed) ---
MONTH_MAP = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}


# --- Helper: Standardize State ---
def clean_state_name(state: str):
    if not isinstance(state, str):
        return state
    st = state.strip().upper()

    # Normalize Johor variants
    if "JOHOR" in st:
        return "JOHOR"

    # Normalize Sarawak variants
    if "SARAWAK" in st:
        return "SARAWAK"

    # Normalize Sabah variants
    if "SABAH" in st:
        return "SABAH"

    # Remove dual language labels (e.g., /WEST JOHORE)
    if "/" in st:
        st = st.split("/")[0].strip()

    return st


# --- MAIN FUNCTION: YEARLY ---
def prepare_yearly(df_land, df_vess):
    df_land = df_land.copy()
    df_vess = df_vess.copy()

    # Required columns check
    required_land = {"Year", "State", "Fish Landing (Tonnes)"}
    for col in required_land:
        if col not in df_land.columns:
            raise ValueError(f"Missing required column in df_land: {col}")

    required_vess = {"Year", "State", "Inboard Powered", "Outboard Powered", "Non-Powered"}
    for col in required_vess:
        if col not in df_vess.columns:
            raise ValueError(f"Missing required column in df_vess: {col}")

    # Standardize State Names
    df_land["State"] = df_land["State"].apply(clean_state_name)
    df_vess["State"] = df_vess["State"].apply(clean_state_name)

    # Aggregate Fish Landing (Year + State)
    yearly_land = df_land.groupby(["Year", "State"])["Fish Landing (Tonnes)"].sum().reset_index()
    yearly_land = yearly_land.rename(columns={"Fish Landing (Tonnes)": "Total Fish Landing (Tonnes)"})

    # Aggregate Vessel counts
    yearly_vess = df_vess.groupby(["Year", "State"])[
        ["Inboard Powered", "Outboard Powered", "Non-Powered"]
    ].sum().reset_index()

    yearly_vess["Total number of fishing vessels"] = (
        yearly_vess["Inboard Powered"].fillna(0) +
        yearly_vess["Outboard Powered"].fillna(0) +
        yearly_vess["Non-Powered"].fillna(0)
    )

    # Merge both
    result = pd.merge(yearly_land, yearly_vess, on=["Year", "State"], how="left")

    result["Total number of fishing vessels"] = result["Total number of fishing vessels"].fillna(0)

    # Sort output
    result = result.sort_values(by=["Year", "State"]).reset_index(drop=True)

    return result


# --- MAIN FUNCTION: MONTHLY ---
def prepare_monthly(df_land, df_vess=None):
    df_land = df_land.copy()

    # Convert Month names → numbers
    if df_land["Month"].dtype == object:
        df_land["Month"] = df_land["Month"].replace(MONTH_MAP)

    # Standardize State
    df_land["State"] = df_land["State"].apply(clean_state_name)

    # Aggregate monthly values
    result = df_land.groupby(["Year", "Month", "State"])["Fish Landing (Tonnes)"].sum().reset_index()

    return result
