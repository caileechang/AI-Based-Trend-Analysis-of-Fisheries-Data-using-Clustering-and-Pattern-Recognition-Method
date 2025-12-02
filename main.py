import streamlit as st
import pandas as pd
import numpy as np
import calendar
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import folium
from streamlit_folium import st_folium
import re
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from difflib import get_close_matches
import time
import plotly.express as px
import plotly.graph_objects as go
import hdbscan


# -----------------------
# INITIALISE default page
# -----------------------
if "plot_option" not in st.session_state:
    st.session_state.plot_option = "🏠 Dashboard Overview"


# Import your clustering modules
#from clustering_method import hierarchical_clustering
@st.cache_data
def load_data():
    url = 'https://www.dropbox.com/scl/fi/4cl5zaor1l32ikyudvf2e/Fisheries-Dataset-vessels-fish-landing.xlsx?rlkey=q2ewpeuzj288ewd17rcqxeuie&st=6h4zijb8&dl=1'
    df_land = pd.read_excel(url, sheet_name='Fish Landing')
    df_vess = pd.read_excel(url, sheet_name='Fish Vessels')

    df_land['Fish Landing (Tonnes)'] = (
        df_land['Fish Landing (Tonnes)']
        .astype(str)
        .str.replace(r'[^\d.]', '', regex=True)
        .replace('', np.nan)
        .astype(float)
    )
    df_land = df_land.dropna(subset=['Fish Landing (Tonnes)']).reset_index(drop=True)
    df_land['Month'] = df_land['Month'].apply(
        lambda x: list(calendar.month_name).index(x.strip().title()) if isinstance(x, str) else x
    )

    for col in ['Inboard Powered', 'Outboard Powered', 'Non-Powered']:
        df_vess[col] = pd.to_numeric(df_vess[col], errors='coerce').fillna(0)
    df_vess['Total number of fishing vessels'] = (
        df_vess['Inboard Powered'] + df_vess['Outboard Powered'] + df_vess['Non-Powered']
    )
    df_vess['State'] = df_vess['State'].str.upper().str.strip()
    df_vess['Year'] = df_vess['Year'].astype(int)

    return df_land, df_vess
    
def prepare_yearly(df_land, df_vess):


    valid_states = [
        "JOHOR TIMUR/EAST JOHORE", "JOHOR BARAT/WEST JOHORE", "JOHOR",
        "MELAKA", "NEGERI SEMBILAN", "SELANGOR", "PAHANG", "TERENGGANU",
        "KELANTAN", "PERAK", "PULAU PINANG", "KEDAH", "PERLIS",
        "SABAH", "SARAWAK", "W.P. LABUAN"
    ]
    valid_states = [s.upper().strip() for s in valid_states]

   
    # CLEAN df_land (FISH LANDING)
   
    land = df_land.copy()
    land['State'] = (
        land['State']
            .astype(str)
            .str.upper()
            .str.replace(r"\s*/\s*", "/", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
    )

    # REMOVE MALAYSIA-level rows BEFORE fuzzy match
    land = land[~land['State'].str.startswith("MALAYSIA")]

    # Fuzzy matching for df_land
    def match_state_land(name):
        matches = get_close_matches(name.upper(), valid_states, n=1, cutoff=0.75)
        return matches[0] if matches else np.nan

    land['State'] = land['State'].apply(match_state_land)
    land = land.dropna(subset=['State'])
    land = land[land['State'].isin(valid_states)]

   
    # GROUP df_land → yearly freshwater/marine totals
   
    yearly_totals = (
        land.groupby(['Year', 'State', 'Type of Fish'])['Fish Landing (Tonnes)']
            .sum()
            .reset_index()
    )

    yearly_pivot = yearly_totals.pivot_table(
        index=['Year', 'State'],
        columns='Type of Fish',
        values='Fish Landing (Tonnes)',
        aggfunc='sum'
    ).reset_index().fillna(0)

    yearly_pivot.columns.name = None
    yearly_pivot.rename(columns={
        'Freshwater': 'Freshwater (Tonnes)',
        'Marine': 'Marine (Tonnes)'
    }, inplace=True)

    yearly_pivot['Total Fish Landing (Tonnes)'] = \
        yearly_pivot.get('Freshwater (Tonnes)', 0) + \
        yearly_pivot.get('Marine (Tonnes)', 0)

    # ======================================================
    # 3) CLEAN df_vess DIRECTLY (no new variable)
    # ======================================================
    df_vess = df_vess.copy()  # overwrite original safely

    df_vess['State'] = (
        df_vess['State']
            .astype(str)
            .str.upper()
            .str.replace(r"\s*/\s*", "/", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
    )

    # REMOVE MALAYSIA-level rows
    df_vess = df_vess[~df_vess['State'].str.startswith("MALAYSIA")]

    # Fuzzy match for df_vess
    def match_state_vess(name):
        matches = get_close_matches(name.upper(), valid_states, n=1, cutoff=0.90)
        return matches[0] if matches else np.nan

    df_vess['State'] = df_vess['State'].apply(match_state_vess)
    df_vess = df_vess.dropna(subset=['State'])
    df_vess = df_vess[df_vess['State'].isin(valid_states)]

    # Clean numeric vessel values
    for col in ['Inboard Powered', 'Outboard Powered', 'Non-Powered']:
        df_vess[col] = pd.to_numeric(df_vess[col], errors='coerce').fillna(0)

    df_vess['Total number of fishing vessels'] = \
        df_vess['Inboard Powered'] + df_vess['Outboard Powered'] + df_vess['Non-Powered']

    df_vess['Year'] = pd.to_numeric(df_vess['Year'], errors='coerce')
    df_vess = df_vess.dropna(subset=['Year'])
    df_vess['Year'] = df_vess['Year'].astype(int)

   
    # MERGE CLEAN df_land + CLEAN df_vess

    merged = pd.merge(
        yearly_pivot,
        df_vess[['State', 'Year', 'Total number of fishing vessels']],
        on=['State', 'Year'],
        how='left'   # IMPORTANT: do NOT use outer join
    ).fillna(0)

    print("LAND:", land.shape, " VESS:", df_vess.shape)
    print("MERGED:", merged.shape)

    return merged.sort_values(['Year', 'State']).reset_index(drop=True)

def prepare_monthly(df_land, df_vess):
    valid_states = [
        "JOHOR TIMUR/EAST JOHORE", "JOHOR BARAT/WEST JOHORE", "JOHOR",
        "MELAKA", "NEGERI SEMBILAN", "SELANGOR", "PAHANG", "TERENGGANU",
        "KELANTAN", "PERAK", "PULAU PINANG", "KEDAH", "PERLIS",
        "SABAH", "SARAWAK", "W.P. LABUAN"
    ]
    valid_states = [s.upper().strip() for s in valid_states]

    land = df_land.copy()
    land["State"] = (
        land["State"]
        .astype(str)
        .str.upper()
        .str.replace(r"\s*/\s*", "/", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    land = land[~land["State"].str.startswith("MALAYSIA")]

    # Fuzzy match states
    def match_state(name):
        matches = get_close_matches(name.upper(), valid_states, n=1, cutoff=0.90)
        return matches[0] if matches else np.nan

    land["State"] = land["State"].apply(match_state)
    land = land.dropna(subset=["State"])
    land = land[land["State"].isin(valid_states)]

    # Convert numeric columns
    land["Month"] = pd.to_numeric(land["Month"], errors="coerce")
    land["Year"] = pd.to_numeric(land["Year"], errors="coerce")
    land["Fish Landing (Tonnes)"] = pd.to_numeric(land["Fish Landing (Tonnes)"], errors="coerce")

    land = land.dropna(subset=["Month", "Year", "Fish Landing (Tonnes)"])

    # Aggregate monthly totals
    monthly_totals = (
        land.groupby(["Year", "Month", "State"], as_index=False)["Fish Landing (Tonnes)"]
        .sum()
    )

    vess = df_vess.copy()
    vess["State"] = (
        vess["State"]
        .astype(str)
        .str.upper()
        .str.replace(r"\s*/\s*", "/", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    vess = vess[~vess["State"].str.startswith("MALAYSIA")]
    vess["State"] = vess["State"].apply(match_state)
    vess = vess.dropna(subset=["State"])
    vess = vess[vess["State"].isin(valid_states)]

    for col in ["Inboard Powered", "Outboard Powered", "Non-Powered"]:
        vess[col] = pd.to_numeric(vess[col], errors="coerce").fillna(0)

    vess["Total number of fishing vessels"] = (
        vess["Inboard Powered"] + vess["Outboard Powered"] + vess["Non-Powered"]
    )

    vess["Year"] = pd.to_numeric(vess["Year"], errors="coerce")
    vess = vess.dropna(subset=["Year"])
    vess["Year"] = vess["Year"].astype(int)

    merged_monthly = pd.merge(
        monthly_totals,
        vess[["State", "Year", "Total number of fishing vessels"]],
        on=["State", "Year"],
        how="left"
    )

    merged_monthly = merged_monthly.sort_values(["Year", "Month", "State"]).reset_index(drop=True)
    return merged_monthly


def evaluate_kmeans_k(data, title_prefix, use_streamlit=True):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import matplotlib.pyplot as plt
    import numpy as np
    import streamlit as st

    silhouette_scores, inertia_scores = [], []
    k_range = range(2, 11)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data)
        sil_score = silhouette_score(data, labels)
        inertia = kmeans.inertia_
        silhouette_scores.append(sil_score)
        inertia_scores.append(inertia)
        print(f"K={k}: Silhouette={sil_score:.4f}, Inertia={inertia:.2f}")

    best_index = np.argmax(silhouette_scores)
    best_k = list(k_range)[best_index]
    best_sil = silhouette_scores[best_index]
    best_inertia = inertia_scores[best_index]

    # --- Plot both side-by-side ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(k_range, silhouette_scores, marker='o')
    axes[0].axvline(best_k, color='red', linestyle='--', label=f"Best k={best_k}")
    axes[0].set_title(f"{title_prefix} - Silhouette Score vs K")
    axes[0].set_xlabel("Number of Clusters (K)")
    axes[0].set_ylabel("Silhouette Score")
    axes[0].legend()

    axes[1].plot(k_range, inertia_scores, marker='o', color='orange')
    axes[1].axvline(best_k, color='red', linestyle='--', label=f"Best k={best_k}")
    axes[1].set_title(f"{title_prefix} - Elbow Method: Inertia vs K")
    axes[1].set_xlabel("Number of Clusters (K)")
    axes[1].set_ylabel("Inertia (WSS)")
    axes[1].legend()

    plt.tight_layout()

    if use_streamlit:
        st.pyplot(fig)
        st.success(f"{title_prefix}: Best k = {best_k} (Silhouette = {best_sil:.3f})")
    else:
        plt.show()
        print(f"\n{title_prefix} - Best k = {best_k} | Silhouette = {best_sil:.4f} | Inertia = {best_inertia:.2f}")

    return best_k, best_sil, best_inertia

def hierarchical_clustering(merged_df):

    import streamlit as st
    import pandas as pd
    import seaborn as sns
    from scipy.cluster.hierarchy import linkage, fcluster
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    import matplotlib.pyplot as plt
    import numpy as np


    # ----------------------------
    # Clean valid states
    # ----------------------------
    valid_states = [
        "JOHOR TIMUR/EAST JOHORE", "JOHOR BARAT/WEST JOHORE", "JOHOR",
        "MELAKA", "NEGERI SEMBILAN", "SELANGOR", "PAHANG", "TERENGGANU",
        "KELANTAN", "PERAK", "PULAU PINANG", "KEDAH", "PERLIS",
        "SABAH", "SARAWAK", "W.P. LABUAN"
    ]

    df = merged_df.copy()

    df["State"] = (
        df["State"]
        .astype(str)
        .str.upper()
        .str.strip()
        .str.replace(r"\s*/\s*", "/", regex=True)
        .str.replace(r"\s+", " ", regex=True)
    )

    df = df[df["State"].isin(valid_states)]

    if df.empty:
        st.warning("No valid states after filtering.")
        return

    # ----------------------------
    # Year selection
    # ----------------------------
    years = sorted(df["Year"].unique())
    selected_year = st.selectbox("Select Year:", years, index=len(years) - 1)

    df_year = df[df["Year"] == selected_year]
    if df_year.empty:
        st.warning("No data for this year.")
        return

    # ----------------------------
    # Group by state averages
    # ----------------------------
    grouped = (
        df_year.groupby("State")[["Total Fish Landing (Tonnes)",
                                  "Total number of fishing vessels"]]
        .mean()
        .reset_index()
    )

    features = ["Total Fish Landing (Tonnes)", "Total number of fishing vessels"]

    # ----------------------------
    # Scaling
    # ----------------------------
    scaler = StandardScaler()
    scaled = scaler.fit_transform(grouped[features])

    # ----------------------------
    # Ward linkage
    # ----------------------------
    Z = linkage(scaled, method="ward")

    # ----------------------------
    # Silhouette Validation (k = 2–6)
    # ----------------------------
    st.markdown("### Silhouette Score Validation")

    cand_k = [2, 3, 4, 5, 6]
    sil_scores = {}

    for k in cand_k:
        # Assign hierarchical clusters
        labels = fcluster(Z, k, criterion="maxclust")
        unique_labels = np.unique(labels)

        # --- VALIDATION: Silhouette requires at least 2 clusters and less than n samples
        if len(unique_labels) < 2 or len(unique_labels) >= len(labels):
            sil_scores[k] = None
            continue

        try:
            sil_scores[k] = silhouette_score(scaled, labels)
        except:
            sil_scores[k] = None

    # --- Filter valid scores only ---
    valid_scores = {k: v for k, v in sil_scores.items() if v is not None}

    if len(valid_scores) == 0:
        st.error("Silhouette cannot be computed for this dataset.")
        return

    best_k = max(valid_scores, key=valid_scores.get)

    # ----------------------------
    # Visualisation Layout
    # ----------------------------
    col1, col2 = st.columns([1, 1])

    # ----------------------------
    # 1️⃣ Silhouette Line Chart
    # ----------------------------
    with col1:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(
            list(valid_scores.keys()),
            list(valid_scores.values()),
            marker="o",
            linewidth=2,
        )
        ax.axvline(best_k, color="red", linestyle="--", label=f"Best k = {best_k}")
        ax.set_xlabel("k")
        ax.set_ylabel("Silhouette Score")
        ax.set_title("Silhouette Scores for k = 2–6")
        ax.legend()
        st.pyplot(fig)

    # ----------------------------
    # 2️⃣ Silhouette Table
    # ----------------------------
    with col2:
        df_sil = (
            pd.DataFrame({
                "k": list(valid_scores.keys()),
                "Silhouette Score": [round(v, 4) for v in valid_scores.values()]
            })
            .sort_values("k")
            .reset_index(drop=True)
        )

        st.dataframe(df_sil, height=230)

    # ----------------------------
    # Best-k Display
    # ----------------------------
    st.success(f"Optimal number of clusters (based on Silhouette): **k = {best_k}**")



    # ----------------------------
    # Final cluster assignment
    # ----------------------------
    grouped["Cluster"] = fcluster(Z, best_k, criterion="maxclust")

    # ----------------------------
    # Seaborn Clustermap (Correct)
    # ----------------------------
    st.markdown("### Fisheries-Based State Grouping Using Hierarchical Clustering")

    df_plot = pd.DataFrame(scaled, columns=["Landing", "Vessels"])
    df_plot["Cluster"] = grouped["Cluster"].values
    df_plot.index = grouped["State"].tolist()

    lut = {
        1: "blue",
        2: "green",
        3: "red",
        4: "purple",
        5: "orange",
        6: "brown"
    }
    row_colors = df_plot["Cluster"].map(lut)

    sns.set_theme(style="white")

    g = sns.clustermap(
        df_plot[["Landing", "Vessels"]],
        method="ward",
        metric="euclidean",
        figsize=(10, 6),
        row_colors=row_colors,
        cmap="viridis",
        dendrogram_ratio=0.2,
        cbar_pos=(0.02, .8, .03, .18)
    )

    with st.container():
        st.pyplot(g.fig)

    # --- FIX: allow Streamlit to continue rendering ---
    st.write("")
    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    # INTERPRETATION CARDS

    st.markdown("## Interpretation of Clusters")

    # Define full-color card backgrounds
    cluster_colors = {
        1: "#1E90FF",   # Blue
        2: "#2ECC71",   # Green
        3: "#E74C3C",   # Red
        4: "#9B59B6",   # Purple
        5: "#F39C12",   # Orange
        6: "#16A085"    # Teal
    }

    real_clusters = sorted(grouped["Cluster"].unique())
    cols = st.columns(len(real_clusters))

    for idx, cid in enumerate(real_clusters):
        subset = grouped[grouped["Cluster"] == cid]
        avg_landing = subset["Total Fish Landing (Tonnes)"].mean()
        avg_vessels = subset["Total number of fishing vessels"].mean()

        # Main color
        base = cluster_colors.get(cid, "#333333")

        # Modern gradient color
        gradient = base + "CC"   # add transparency for smooth effect

        card_html = f"""
        <div style="
            background: linear-gradient(145deg, {base}, {gradient});
            border-radius: 20px;
            padding: 28px;
            color: white;
            box-shadow: 0 6px 22px rgba(0,0,0,0.35);
            min-height: 260px;
        ">
        <h2 style="text-align:center; margin-top:0; font-size:28px; font-weight:700;">
            Cluster {cid}
        </h2>

        <p style="font-size:18px; line-height:1.5;">
            <b>Avg landing:</b> {avg_landing:,.2f} tonnes
        </p>

        <p style="font-size:18px; line-height:1.5;">
            <b>Avg vessels:</b> {avg_vessels:,.0f}
        </p>

        <p style="font-size:18px; line-height:1.5; margin-bottom:4px;">
            <b>States:</b>
        </p>

        <p style="font-size:17px; opacity:0.95;">
            {", ".join(subset["State"].tolist())}
        </p>

        </div>
        """
        cols[idx].markdown(card_html, unsafe_allow_html=True)

    # ----------------------------
    # Final table
    # ----------------------------
    st.markdown("### Cluster Assignments (TRUE Hierarchical Clusters)")
    st.dataframe(
        grouped[["State",
                 "Total Fish Landing (Tonnes)",
                 "Total number of fishing vessels",
                 "Cluster"]]
        .sort_values("Cluster")
        .reset_index(drop=True)
    )

    
def main():
    
   
    import pandas as pd

    st.set_page_config(layout='wide')
     # ======================================
    # GLOBAL PREMIUM CSS (NEUMORPHISM + ANIMATION)
    # ======================================
    st.markdown("""
    <style>

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .neu-card {
        background: #1b1b1b;
        border-radius: 24px;
        padding: 28px;
        margin-bottom: 20px;
        border: 1px solid rgba(255,255,255,0.06);

        /* NEUMORPHISM SHADOW */
        box-shadow:
            9px 9px 20px rgba(0,0,0,0.55),
            -9px -9px 20px rgba(255,255,255,0.04);

        animation: fadeIn 0.55s ease-out;
        transition: all 0.25s ease;
        position: relative;
        overflow: hidden;
    }

    /* HOVER EFFECT */
    .neu-card:hover {
        transform: translateY(-6px);
        box-shadow:
            12px 12px 28px rgba(0,0,0,0.65),
            -12px -12px 28px rgba(255,255,255,0.06);
    }

    /* SHIMMER HIGHLIGHT */
    .shimmer {
        background: linear-gradient(
            90deg,
            rgba(255,255,255,0) 0%,
            rgba(255,255,255,0.15) 50%,
            rgba(255,255,255,0) 100%
        );
        position: absolute;
        top:0; left:0;
        height:100%; width:100%;
        transform: translateX(-100%);
        animation: shimmerMove 2.7s infinite;
    }

    @keyframes shimmerMove {
        0%   { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }

    </style>
    """, unsafe_allow_html=True)

    #st.title("Fisheries Clustering & Pattern Recognition Dashboard")

   

    # --- Load base data or use newly merged uploaded data ---
    if "base_land" not in st.session_state:
        st.session_state.base_land, st.session_state.base_vess = load_data()
        st.session_state.data_updated = False  # no uploaded data yet
    
    if "data_updated" in st.session_state and st.session_state.data_updated:
        df_land = st.session_state.base_land
        df_vess = st.session_state.base_vess
    else:
        df_land = st.session_state.base_land
        df_vess = st.session_state.base_vess


    # Upload additional yearly CSV
    st.sidebar.markdown("### Upload Your Yearly Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload Excel file only (.xlsx)", type=["xlsx"])

    if uploaded_file:
            try:
                excel_data = pd.ExcelFile(uploaded_file)
                sheet_names = [s.lower() for s in excel_data.sheet_names]
        
                if "fish landing" in sheet_names and "fish vessels" in sheet_names:
                    user_land = pd.read_excel(excel_data, sheet_name="Fish Landing")
                    user_vess = pd.read_excel(excel_data, sheet_name="Fish Vessels")
                else:
                    st.warning(" The uploaded file must contain sheets named 'Fish Landing' and 'Fish Vessels'.")
                    user_land, user_vess = None, None
        
                if user_land is not None:
                    #st.subheader("New dataset uploaded")
                    #st.dataframe(user_land, use_container_width=True, height=400)
                    msg2=st.info(f"Detected uploaded years: {sorted(user_land['Year'].dropna().unique().astype(int).tolist())}")
                   
        
                    # --- Clean uploaded data to match base format ---
                    user_land.columns = user_land.columns.str.strip().str.title()
                    user_land['Month'] = user_land['Month'].astype(str).str.strip().str.title()
                    user_land['State'] = user_land['State'].astype(str).str.upper().str.strip()
                    user_land['Type Of Fish'] = user_land['Type Of Fish'].astype(str).str.title().str.strip()
                    user_land.rename(columns={'Type Of Fish': 'Type of Fish'}, inplace=True)

                     # Convert month names to numbers
                    month_map = {
                        'January': 1, 'Jan': 1, 'February': 2, 'Feb': 2, 'March': 3, 'Mar': 3,
                        'April': 4, 'Apr': 4, 'May': 5, 'June': 6, 'Jun': 6, 'July': 7, 'Jul': 7,
                        'August': 8, 'Aug': 8, 'September': 9, 'Sep': 9, 'October': 10, 'Oct': 10,
                        'November': 11, 'Nov': 11, 'December': 12, 'Dec': 12
                    }
                    user_land['Month'] = user_land['Month'].map(month_map).fillna(user_land['Month'])
                    user_land['Month'] = pd.to_numeric(user_land['Month'], errors='coerce')
        
                    # Ensure numeric types
                    user_land['Year'] = pd.to_numeric(user_land['Year'], errors='coerce')
                    user_land['Fish Landing (Tonnes)'] = pd.to_numeric(user_land['Fish Landing (Tonnes)'], errors='coerce')
                    user_land.dropna(subset=['Year', 'Fish Landing (Tonnes)', 'State', 'Type of Fish'], inplace=True)
        
                    # --- Merge uploaded data with base historical data (SAME structure) ---
                    df_land = pd.concat([df_land, user_land], ignore_index=True).drop_duplicates(subset=['State', 'Year', 'Month', 'Type of Fish'])
                   
                    msg1=st.toast(" Uploaded data successfully merged with existing dataset.")
                    
                    # --- Clean uploaded vessel data to match base format ---
                    user_vess.columns = user_vess.columns.str.strip().str.title()
                    user_vess['State'] = user_vess['State'].astype(str).str.upper().str.strip()
                    
                    for col in ['Inboard Powered', 'Outboard Powered', 'Non-Powered']:
                        user_vess[col] = pd.to_numeric(user_vess[col], errors='coerce').fillna(0)
                    
                    user_vess['Total number of fishing vessels'] = (
                        user_vess['Inboard Powered'] +
                        user_vess['Outboard Powered'] +
                        user_vess['Non-Powered']
                    )
                    
                    user_vess['Year'] = pd.to_numeric(user_vess['Year'], errors='coerce')
                    user_vess = user_vess.dropna(subset=['Year'])
                    user_vess['Year'] = user_vess['Year'].astype(int)

                    df_vess = pd.concat([df_vess, user_vess], ignore_index=True).drop_duplicates(subset=['State', 'Year'])

                                        # Update session state immediately and keep merged data
                    st.session_state.base_land = df_land.copy()
                    st.session_state.base_vess = df_vess.copy()
                    st.session_state.data_updated = True  # mark that new data exists
                    st.cache_data.clear()
                    st.sidebar.success("New dataset merged. Visualizations will refresh automatically.")

        
            except Exception as e:
                st.error(f"Error reading uploaded file: {e}")

    merged_df = prepare_yearly(df_land, df_vess)
    

    merged_monthly = prepare_monthly(df_land, df_vess)


    sidebar_options = [
        "🏠 Dashboard Overview",
        "Yearly Fish Landing Summary",
        "Optimal K for Monthly & Yearly",
        "Yearly Cluster Trends for Marine and Freshwater Fish",
        "2D KMeans Scatter",
        "3D KMeans Clustering",
        "Automatic DBSCAN",
        "Unified HDBSCAN Outlier Detection",
        "HDBSCAN",
        "HDBSCAN Outlier Detection",
        "Hierarchical Clustering",
        "Geospatial Map",
        "Interactive Geospatial Map",
        "Geospatial Map(Heatmap)",
        "Geospatial Map (Upgraded)",
        "Geospatial Map (Upgraded)2"
    ]

    # Ensure default page exists
    if "plot_option" not in st.session_state:
        st.session_state.plot_option = "🏠 Dashboard Overview"

    # Show sidebar radio with correct selected index
    selected = st.sidebar.radio(
        "Choose a visualization:",
        sidebar_options,
        index=sidebar_options.index(st.session_state.plot_option)
    )

    # Only update session state IF user clicked sidebar
    if selected != st.session_state.plot_option:
        st.session_state.plot_option = selected
        st.rerun()

    # 🔥 THIS line creates your local variable
    plot_option = st.session_state.plot_option


    # ============================================================
    # 2) DASHBOARD OVERVIEW PAGE (AUTO-GENERATED CARDS)
    # ============================================================

    if plot_option == "🏠 Dashboard Overview":

        st.markdown("## 📊 Fisheries Analytics Dashboard Overview")
        st.markdown(
            "<p style='color:#bbb'>Click any card below to open the full visualisation.</p>",
            unsafe_allow_html=True
        )

        # All visualisations EXCEPT the first "Home"
        visualisation_pages = sidebar_options[1:]

        # Icons + Colors
        icons = ["📘","📊","🌊","🟦","🧊","🌐","⚡","🌳",
                "🗺️","🖥️","🔥","✨","📍","📡","🔍"]

        colors = [
            "#00E5FF","#7DFFAF","#FFA07A","#81C7F5","#DDA0DD",
            "#FFD700","#FF7F7F","#8FBC8F","#00CED1","#4B7BE5",
            "#F7A440","#C15AFF","#57E39A","#E33F5F","#A4E5FF"
        ]

        cols = st.columns(3)

        for i, name in enumerate(visualisation_pages):

            icon = icons[i % len(icons)]
            color = colors[i % len(colors)]

            with cols[i % 3]:

                    # Clickable button styled as transparent over the card
                if st.button(
                    label=f"{icon} {name}", 
                    key=f"card_{i}", 
                    help=f"Open {name}", 
                    use_container_width=True
                ):
                    st.session_state.plot_option = name
                    st.rerun()

                # Inject CSS to style the button as a card
                st.markdown(
                    f"""
                    <style>
                    div[data-testid="stButton"][key="card_{i}"] > button {{
                        height: 140px;
                        text-align: left;
                        padding: 20px;
                        border-radius: 14px;
                        font-size: 18px;
                        color: white;
                        background: linear-gradient(135deg, {color}33, #111);
                        border: 1px solid {color}55;
                        box-shadow: 0 0 12px {color}33;
                    }}
                    div[data-testid="stButton"][key="card_{i}"] > button:hover {{
                        border: 1px solid {color};
                        transform: scale(1.01);
                        transition: 0.1s ease-in-out;
                    }}
                    </style>
                    """,
                    unsafe_allow_html=True,
                )




    elif plot_option == "Monthly Trends by Cluster":
       # monthly = df_land.groupby(['Year', 'Month'])['Fish Landing (Tonnes)'].sum().reset_index()
       
        # --- Use merged dataset (always latest) ---
        monthly = st.session_state.base_land.groupby(['Year', 'Month'])['Fish Landing (Tonnes)'].sum().reset_index()
                # --- Ensure Year/Month are numeric ---
        monthly['Year'] = pd.to_numeric(monthly['Year'], errors='coerce')
        monthly['Month'] = pd.to_numeric(monthly['Month'], errors='coerce')
        # --- Dynamically filter realistic range ---
        latest_year = int(monthly['Year'].max())
        monthly = monthly[
            (monthly['Year'].between(2000, latest_year)) &
            (monthly['Month'].between(1, 12))
        ]
        # --- Convert to datetime safely ---
        monthly['MonthYear'] = pd.to_datetime(
            monthly['Year'].astype(int).astype(str) + '-' + monthly['Month'].astype(int).astype(str) + '-01',
            errors='coerce'
        )
        monthly = monthly.dropna(subset=['MonthYear'])

        #monthly['MonthYear'] = pd.to_datetime(monthly['Year'].astype(str) + '-' + monthly['Month'].astype(str).str.zfill(2))
        X = StandardScaler().fit_transform(monthly[['Month', 'Fish Landing (Tonnes)']])
        monthly['Cluster'] = KMeans(n_clusters=3, random_state=42).fit_predict(X)

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=monthly.sort_values('MonthYear'), x='MonthYear', y='Fish Landing (Tonnes)', hue='Cluster', marker='o', ax=ax, sort=False, linewidth=1.5, style='Cluster')

        #sns.lineplot(data=monthly.sort_values('MonthYear'), x='MonthYear', y='Fish Landing (Tonnes)', hue='Cluster', marker='o', ax=ax)
        ax.set_title("Monthly Fish Landing Trends by Cluster")
        ax.set_xlabel("Month-Year")
        ax.set_ylabel("Fish Landing (Tonnes)")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    elif plot_option == "Yearly Fish Landing Summary":
           
        import seaborn as sns
        import matplotlib.pyplot as plt
        import numpy as np

        st.markdown("##  Yearly Fish Landing")
         
        

        # --- ALWAYS use cleaned yearly summary from prepare_yearly ---
        if uploaded_file:
            yearly_summary = prepare_yearly(df_land, df_vess)
        else:
            yearly_summary = st.session_state.get(
                "yearly_summary", prepare_yearly(df_land, df_vess)
            )

        st.session_state.yearly_summary = yearly_summary

        # ------------------------------------------------------
        # A) Summary Cards + Lollipop FIRST
        # ------------------------------------------------------

        # Get latest year
        latest_year = int(yearly_summary["Year"].max())
        prev_year = latest_year - 1

        filtered_latest = yearly_summary[yearly_summary["Year"] == latest_year].copy()

        # Sort by landing
        sorted_desc = filtered_latest.sort_values(
            "Total Fish Landing (Tonnes)", ascending=False
        )
        top3 = sorted_desc.head(3).copy()

        # Previous year values
        def get_prev(state):
            prev = yearly_summary[
                (yearly_summary["Year"] == prev_year)
                & (yearly_summary["State"] == state)
            ]
            if prev.empty:
                return np.nan
            return prev["Total Fish Landing (Tonnes)"].iloc[0]

        top3["Prev_Year"] = top3["State"].apply(get_prev)

        def growth_text(curr, prev):
            # SAFELY handle missing or invalid previous-year values
            try:
                if prev is None or prev == "" or float(prev) == 0:
                    return "<span style='color:#888;'>No comparison</span>"
            except:
                return "<span style='color:#888;'>No comparison</span>"

            # Convert safely to float
            prev = float(prev)

            # Compute percentage change
            change = (curr - prev) / prev * 100
            arrow = "↑" if change >= 0 else "↓"
            color = "#4CAF50" if change >= 0 else "#ff4d4d"

            # Label previous year if provided
            label = f" vs {prev_year}" if prev_year else " vs previous"

            return f"<span style='color:{color}; font-size:16px;'>{arrow} {change:.1f}%{label}</span>"
        medal_colors = ["#FFD700", "#C0C0C0", "#CD7F32"]

        
        st.markdown(f"### 🏅 Top 3 States in {latest_year}")

        card_cols = st.columns(3)
        
        
        for idx, (_, row) in enumerate(top3.iterrows()):
            with card_cols[idx]:
                state = row["State"]
                total = row["Total Fish Landing (Tonnes)"]
                prev_val = row["Prev_Year"]
                growth_html = growth_text(total, prev_val)

                card_html = f"""
                <div style="
                    background: radial-gradient(circle at top left, rgba(0,255,255,0.25), rgba(0,0,0,0.9));
                    border-radius: 14px;
                    padding: 18px 18px 14px 18px;
                    border: 1px solid rgba(0,255,255,0.35);
                    box-shadow: 0 0 18px rgba(0,255,255,0.18);
                    min-height: 150px;
                ">
                    <div style="font-size:18px; color:'white'; margin-bottom:6px;">
                        <span style="color:{medal_colors[idx]}; font-size:22px;">●</span>
                        <b style="color:white; margin-left:6px;">#{idx+1} {state}</b>
                    </div>
                    <div style="font-size:30px; color:white; font-weight:bold;">
                        {total:,.0f} <span style="font-size:16px; color:#bbb;">tonnes</span>
                    </div>
                    <div style="margin-top:8px;">
                        {growth_html}
                    </div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)

        st.markdown("---")


        # ------------------------------------------------------
        # CHART FOR LATEST YEAR
        # ------------------------------------------------------
        st.markdown(f"### Total Fish Landing by State ({latest_year})")

        filtered_sorted = filtered_latest.sort_values(
            "Total Fish Landing (Tonnes)", ascending=True
        )

        import plotly.graph_objects as go
        fig = go.Figure()

        # Stem lines
        fig.add_trace(
            go.Scatter(
                x=filtered_sorted["Total Fish Landing (Tonnes)"],
                y=filtered_sorted["State"],
                mode="lines",
                line=dict(color="rgba(0,255,255,0.3)", width=3),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        # Neon markers
        fig.add_trace(
            go.Scatter(
                x=filtered_sorted["Total Fish Landing (Tonnes)"],
                y=filtered_sorted["State"],
                mode="markers+text",
                marker=dict(color="#00E5FF", size=11, line=dict(color="white", width=1)),
                text=[f"{v:,.0f}" for v in filtered_sorted["Total Fish Landing (Tonnes)"]],
                textposition="middle right",
                textfont=dict(color="white", size=11),
                hovertemplate="State: %{y}<br>Landing: %{x:,.0f}<extra></extra>",
                showlegend=False,
            )
        )

        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                title="Total Fish Landing (Tonnes)",
                gridcolor="rgba(255,255,255,0.08)",
            ),
            yaxis=dict(title="", categoryorder="array", categoryarray=filtered_sorted["State"]),
            margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # ------------------------------------------------------
        # C) NOW SHOW YEAR SELECTOR & TABLE
        # ------------------------------------------------------
        st.markdown("### Select a Year to View Full Details")

        selected_year = st.selectbox(
            "Choose a year:",
            sorted(yearly_summary["Year"].unique()),
            index=len(yearly_summary["Year"].unique()) - 1,
        )

        filtered_selected = yearly_summary[
            yearly_summary["Year"] == selected_year
        ].sort_values("Total Fish Landing (Tonnes)", ascending=False)

        st.markdown(f"### Fish Landing by State — {selected_year}")
        st.dataframe(filtered_selected, use_container_width=True, height=350)



    elif plot_option == "Yearly Cluster Trends for Marine and Freshwater Fish":

        import matplotlib.pyplot as plt
        import seaborn as sns

        # ======================================
        # GLOBAL CARD STYLE + CHART STYLES
        # ======================================
        card_style = """
            background-color: #1e1e1e;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #444;
            margin-bottom: 20px;
        """

        colors = {
            "Freshwater (Tonnes)": "tab:blue",
            "Marine (Tonnes)": "tab:red"
        }

        markers = {
            "Freshwater (Tonnes)": "o",
            "Marine (Tonnes)": "^"
        }

        linestyles = ["solid", "dashed", "dotted", "dashdot"]

        # ======================================
        # PAGE HEADER
        # ======================================
        st.markdown("""
            <h2 style='color:white;'>🎣 Fish Landing Trends (Cluster Analysis)</h2>
            <p style='color:#ccc; margin-top:-10px;'>
                Compare freshwater & marine fish landings across yearly or monthly periods using K-Means cluster grouping.
            </p>
        """, unsafe_allow_html=True)

        st.markdown("<hr style='border:0.5px solid #444;'>", unsafe_allow_html=True)

        # ======================================
        # OPTIONS SECTION
        # ======================================
        with st.container():

            st.markdown(
                "<p style='color:#ccc; margin-top:-12px; font-size:15px;'>"
                "Please select the period and trend to display the fish landing analysis."
                "</p>",
                unsafe_allow_html=True
            )

            opt_col1, opt_col2 = st.columns([1, 2])

            with opt_col1:
                period_choice = st.radio("Period:", ["Yearly", "Monthly"], horizontal=True)

            with opt_col2:
                trend_option = st.radio(
                    "Trend:", ["Freshwater", "Marine", "Both"], horizontal=True
                )

        # ======================================
        # YEARLY VIEW
        # ======================================
        if period_choice == "Yearly":

            yearly = (
                df_land.groupby(["Year", "Type of Fish"])["Fish Landing (Tonnes)"]
                .sum()
                .reset_index()
                .pivot(index="Year", columns="Type of Fish",
                    values="Fish Landing (Tonnes)")
                .fillna(0)
                .reset_index()
            )

            yearly.rename(columns={
                "Freshwater": "Freshwater (Tonnes)",
                "Marine": "Marine (Tonnes)"
            }, inplace=True)

            latest_year = yearly["Year"].max()
            prev_year = latest_year - 1

            def safe_get(df, year, col):
                row = df.loc[df["Year"] == year, col]
                return row.values[0] if len(row) else 0

            def growth_html(curr, prev):
                try:
                    prev = float(prev)
                    curr = float(curr)
                except:
                    return "<span style='color:gray;'>–</span>"

                if prev == 0:
                    return "<span style='color:gray;'>–</span>"

                ratio = curr / prev
                diff = curr - prev

                if ratio >= 1:
                    color = "lightgreen"
                    arrow = "↑"
                    word = "increased"
                else:
                    color = "#ff4d4d"
                    arrow = "↓"
                    word = "decreased"

                return (
                    f"<span style='color:{color}; font-size:18px;'>"
                    f"{arrow} {ratio:.2f}x • {word} by <b>{abs(diff):,.0f}</b> tonnes"
                    "</span>"
                )

            fw_latest = safe_get(yearly, latest_year, "Freshwater (Tonnes)")
            fw_prev = safe_get(yearly, prev_year, "Freshwater (Tonnes)")
            ma_latest = safe_get(yearly, latest_year, "Marine (Tonnes)")
            ma_prev = safe_get(yearly, prev_year, "Marine (Tonnes)")

            # Premium gradient card
            card_style = """
                background: linear-gradient(135deg, #06373d 0%, #001f24 100%);
                padding: 30px 35px;
                border-radius: 20px;
                border: 1.2px solid rgba(0, 255, 200, 0.25);
                box-shadow: 0 0 18px rgba(0, 255, 200, 0.12);
                transition: all 0.25s ease;
            """

            st.markdown("""
                <style>
                .card:hover {
                    transform: translateY(-5px);
                    box-shadow: 0 0 25px rgba(0,255,200,0.25);
                }
                </style>
            """, unsafe_allow_html=True)

            st.markdown(f"## Landing Summary in {latest_year}")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    f"""
                    <div style="{card_style}">
                        <h3 style="color:white;">Freshwater Landing</h3>
                        <h1 style="color:white; font-size:42px;"><b>{fw_latest:,.0f}</b> tonnes</h1>
                        {growth_html(fw_latest, fw_prev)}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    f"""
                    <div style="{card_style}">
                        <h3 style="color:white;">Marine Landing</h3>
                        <h1 style="color:white; font-size:42px;"><b>{ma_latest:,.0f}</b> tonnes</h1>
                        {growth_html(ma_latest, ma_prev)}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            # ======================================
            # YEARLY CLUSTER PLOT
            # ======================================
            features = ["Freshwater (Tonnes)", "Marine (Tonnes)"]
            scaled = StandardScaler().fit_transform(yearly[features])
            best_k = st.session_state.get("best_k_yearly", 3)

            yearly["Cluster"] = KMeans(n_clusters=best_k, random_state=42).fit_predict(scaled)

            st.markdown(f"**Optimal clusters used:** {best_k}")

            melted = yearly.melt(
                id_vars=["Year", "Cluster"],
                value_vars=["Freshwater (Tonnes)", "Marine (Tonnes)"],
                var_name="Type",
                value_name="Landing",
            )

            fig, ax = plt.subplots(figsize=(14, 6))

            for fish_type in ["Freshwater (Tonnes)", "Marine (Tonnes)"]:

                show_this = (trend_option == "Both"
                            or trend_option.lower() in fish_type.lower())

                if show_this:
                    for cl in sorted(melted["Cluster"].unique()):
                        subset = melted[
                            (melted["Type"] == fish_type)
                            & (melted["Cluster"] == cl)
                        ]

                        sns.lineplot(
                            data=subset,
                            x="Year",
                            y="Landing",
                            color=colors[fish_type],
                            linestyle=linestyles[cl % len(linestyles)],
                            marker=markers[fish_type],
                            ax=ax,
                            label=f"{fish_type.replace('(Tonnes)','')} – Cluster {cl}",
                        )

            ax.set_title(f"Yearly Fish Landing Trends (k={best_k})")
            ax.set_ylabel("Landing (Tonnes)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)

            st.pyplot(fig)

        # ======================================
        # MONTHLY VIEW
        # ======================================
        else:

            monthly = (
                df_land.groupby(["Year", "Month", "Type of Fish"])["Fish Landing (Tonnes)"]
                .sum()
                .reset_index()
                .pivot(index=["Year", "Month"], columns="Type of Fish",
                    values="Fish Landing (Tonnes)")
                .fillna(0)
                .reset_index()
            )

            monthly.rename(columns={
                "Freshwater": "Freshwater (Tonnes)",
                "Marine": "Marine (Tonnes)"
            }, inplace=True)

            monthly["MonthYear"] = pd.to_datetime(
                monthly["Year"].astype(str) + "-" +
                monthly["Month"].astype(str) + "-01"
            )

            latest_date = monthly["MonthYear"].max()
            prev_date = latest_date - pd.DateOffset(months=1)

            def safe_month_value(df, date, col):
                v = df.loc[df["MonthYear"] == date, col]
                return v.values[0] if len(v) else 0

            def calc_growth_month_html(curr, prev):
                if prev == 0:
                    return "<span style='color:gray'>–</span>"
                ratio = curr / prev
                if ratio >= 1:
                    return f"<span style='color:lightgreen'>↑ {ratio:.2f}x</span>"
                else:
                    return f"<span style='color:#ff4d4d'>↓ {ratio:.2f}x</span>"

            fw = safe_month_value(monthly, latest_date, "Freshwater (Tonnes)")
            fw_prev = safe_month_value(monthly, prev_date, "Freshwater (Tonnes)")
            ma = safe_month_value(monthly, latest_date, "Marine (Tonnes)")
            ma_prev = safe_month_value(monthly, prev_date, "Marine (Tonnes)")

            st.markdown(f"## Landing Summary in {latest_date.strftime('%B %Y')}")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    f"""
                    <div style="{card_style}">
                        <h3 style="color:white;">Freshwater Landing</h3>
                        <h1 style="color:white; font-size:42px;"><b>{fw:,.0f}</b> tonnes</h1>
                        {calc_growth_month_html(fw, fw_prev)}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    f"""
                    <div style="{card_style}">
                        <h3 style="color:white;">Marine Landing</h3>
                        <h1 style="color:white; font-size:42px;"><b>{ma:,.0f}</b> tonnes</h1>
                        {calc_growth_month_html(ma, ma_prev)}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            # ======================================
            # MONTHLY CLUSTER PLOT
            # ======================================
            features = ["Freshwater (Tonnes)", "Marine (Tonnes)"]
            scaled = StandardScaler().fit_transform(monthly[features])
            best_k = st.session_state.get("best_k_monthly", 3)

            monthly["Cluster"] = KMeans(n_clusters=best_k, random_state=42).fit_predict(
                scaled
            )

            st.markdown(f"**Optimal clusters used:** {best_k}")

            melted = monthly.melt(
                id_vars=["MonthYear", "Cluster"],
                value_vars=["Freshwater (Tonnes)", "Marine (Tonnes)"],
                var_name="Type",
                value_name="Landing",
            )

            fig, ax = plt.subplots(figsize=(14, 6))

            for fish_type in ["Freshwater (Tonnes)", "Marine (Tonnes)"]:

                show_this = (trend_option == "Both"
                            or trend_option.lower() in fish_type.lower())

                if show_this:
                    for cl in sorted(melted["Cluster"].unique()):
                        subset = melted[
                            (melted["Type"] == fish_type)
                            & (melted["Cluster"] == cl)
                        ]

                        sns.lineplot(
                            data=subset,
                            x="MonthYear",
                            y="Landing",
                            color=colors[fish_type],
                            linestyle=linestyles[cl % len(linestyles)],
                            marker=markers[fish_type],
                            ax=ax,
                            label=f"{fish_type.replace('(Tonnes)', '')} – Cluster {cl}",
                        )

            plt.xticks(rotation=45)
            ax.set_title(f"Monthly Fish Landing Trends (k={best_k})")
            ax.set_ylabel("Landing (Tonnes)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)

            st.pyplot(fig)

   

    elif plot_option == "Optimal K for Monthly & Yearly":

        import pandas as pd

        st.subheader("Determination of Optimal K")

        # ------------------------------------------------------------
        # Initialize safe defaults
        # ------------------------------------------------------------
        best_k_monthly = None
        best_sil_monthly = None
        best_k_yearly = None
        best_sil_yearly = None

        # ------------------------------------------------------------
        # Helper: Normalize column names
        # ------------------------------------------------------------
        def normalize_columns(df):
            df.columns = (
                df.columns
                .str.strip()
                .str.lower()
                .str.replace(" ", "")
                .str.replace("(tonnes)", "")
            )
            return df

        # ============================================================
        # 1️⃣  MONTHLY COMPOSITION
        # ============================================================
        st.markdown("### 📘 Monthly Fish Landing Composition")

        try:
            monthly_comp = (
                df_land.groupby(['Year', 'Month', 'Type of Fish'])['Fish Landing (Tonnes)']
                .sum()
                .reset_index()
                .pivot_table(
                    index=['Year', 'Month'],
                    columns='Type of Fish',
                    values='Fish Landing (Tonnes)',
                    aggfunc='sum'
                )
                .fillna(0)
                .reset_index()
            )

            # Normalize column names
            monthly_comp = normalize_columns(monthly_comp)

            # Rename safely
            monthly_comp.rename(columns={
                'freshwater': 'Freshwater (Tonnes)',
                'marine': 'Marine (Tonnes)'
            }, inplace=True)

            # Check required columns
            if 'Freshwater (Tonnes)' not in monthly_comp.columns or 'Marine (Tonnes)' not in monthly_comp.columns:
                st.error("❌ Monthly dataset is missing Freshwater or Marine category.")
            else:
                # Scale
                scaled_monthly = StandardScaler().fit_transform(
                    monthly_comp[['Freshwater (Tonnes)', 'Marine (Tonnes)']]
                )

                # Evaluate best k
                (
                    best_k_monthly,
                    best_sil_monthly,
                    _inertia_m
                ) = evaluate_kmeans_k(
                    scaled_monthly,
                    "Monthly Fish Landing ",
                    use_streamlit=True
                )

        except Exception as e:
            st.error(f"❌ Error in monthly computation: {e}")

        # ============================================================
        # 2️⃣  YEARLY COMPOSITION
        # ============================================================
        st.markdown("### 📗 Yearly Fish Landing Composition")

        try:
            yearly_comp = (
                df_land.groupby(['Year', 'Type of Fish'])['Fish Landing (Tonnes)']
                .sum()
                .reset_index()
                .pivot_table(
                    index='Year',
                    columns='Type of Fish',
                    values='Fish Landing (Tonnes)',
                    aggfunc='sum'
                )
                .fillna(0)
                .reset_index()
            )

            yearly_comp = normalize_columns(yearly_comp)

            yearly_comp.rename(columns={
                'freshwater': 'Freshwater (Tonnes)',
                'marine': 'Marine (Tonnes)'
            }, inplace=True)

            if 'Freshwater (Tonnes)' not in yearly_comp.columns or 'Marine (Tonnes)' not in yearly_comp.columns:
                st.error("❌ Yearly dataset is missing Freshwater or Marine category.")
            else:
                scaled_yearly = StandardScaler().fit_transform(
                    yearly_comp[['Freshwater (Tonnes)', 'Marine (Tonnes)']]
                )

                (
                    best_k_yearly,
                    best_sil_yearly,
                    _inertia_y
                ) = evaluate_kmeans_k(
                    scaled_yearly,
                    "Yearly Fish Landing ",
                    use_streamlit=True
                )

        except Exception as e:
            st.error(f"❌ Error in yearly computation: {e}")

        # ============================================================
        # 3️⃣  SUMMARY TABLE
        # ============================================================
        st.markdown("### 🧾 Summary of Optimal K Results")

        summary = []

        if best_k_monthly is not None:
            summary.append([
                "Monthly (Freshwater + Marine)",
                best_k_monthly,
                f"{best_sil_monthly:.3f}" if best_sil_monthly else "N/A"
            ])

        if best_k_yearly is not None:
            summary.append([
                "Yearly (Freshwater + Marine)",
                best_k_yearly,
                f"{best_sil_yearly:.3f}" if best_sil_yearly else "N/A"
            ])

        if len(summary) == 0:
            st.warning("⚠ No valid results available.")
        else:
            st.table(pd.DataFrame(
                summary,
                columns=["Dataset", "Best K", "Silhouette Score"]
            ))

        # ============================================================
        # 4️⃣  SAVE STATE FOR OTHER PAGES (Critical)
        # ============================================================
        if best_k_monthly is not None:
            st.session_state['best_k_monthly'] = best_k_monthly

        if best_k_yearly is not None:
            st.session_state['best_k_yearly'] = best_k_yearly


    elif plot_option == "2D KMeans Scatter":
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        st.subheader("Automatic 2D K-Means Clustering (with Elbow & Silhouette Analysis)")
    
        # --- Step 1: Prepare data ---
        features = merged_df[['Total Fish Landing (Tonnes)', 'Total number of fishing vessels']]
        scaled = StandardScaler().fit_transform(features)
    
        # --- Step 2: Compute inertia (Elbow) and silhouette for k = 2–10 ---
        ks = range(2, 11)
        inertia = []
        silhouette = []
    
        for k in ks:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(scaled)
            inertia.append(kmeans.inertia_)
            sil = silhouette_score(scaled, labels)
            silhouette.append(sil)
    
        # --- Step 3: Determine the best k (highest silhouette) ---
        best_k = ks[np.argmax(silhouette)]
    
        # --- Step 4: Plot both metrics side by side ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
        # Elbow plot
        ax1.plot(ks, inertia, marker='o')
        ax1.set_title("Elbow Method")
        ax1.set_xlabel("k")
        ax1.set_ylabel("Inertia")
        ax1.axvline(best_k, color='red', linestyle='--', label=f"Best k = {best_k}")
        ax1.legend()
    
        # Silhouette plot
        ax2.plot(ks, silhouette, marker='o', color='orange')
        ax2.set_title("Silhouette Score")
        ax2.set_xlabel("k")
        ax2.set_ylabel("Score")
        ax2.axvline(best_k, color='red', linestyle='--', label=f"Best k = {best_k}")
        ax2.legend()
    
        st.pyplot(fig)
    
        # --- Step 5: Fit the final model using best_k ---
        final_model = KMeans(n_clusters=best_k, random_state=42)
        merged_df['Cluster'] = final_model.fit_predict(scaled)
    
        # --- Step 6: Display summary ---
        st.success(f"Optimal number of clusters automatically determined: **k = {best_k}**")
        st.markdown("Clusters below are determined automatically based on the **highest Silhouette Score** and Elbow consistency.")
    
        # --- Step 7: Show 2D scatter ---
        fig2, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=merged_df,
            x='Total number of fishing vessels',
            y='Total Fish Landing (Tonnes)',
            hue='Cluster',
            palette='viridis',
            s=70,
            ax=ax
        )
        ax.set_title(f"Automatic 2D K-Means Clustering (k={best_k})")
        st.pyplot(fig2)
    


   

    elif plot_option == "3D KMeans Clustering":
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        from mpl_toolkits.mplot3d import Axes3D

        st.subheader("Automatic 3D K-Means Clustering")

        # ---------------------------------------------------
        # STEP 1: PREPARE DATA
        # ---------------------------------------------------
        features = merged_df[['Total Fish Landing (Tonnes)', 'Total number of fishing vessels']]
        scaled = StandardScaler().fit_transform(features)

        # ---------------------------------------------------
        # STEP 2: AUTOMATICALLY FIND BEST k (Silhouette)
        # ---------------------------------------------------
        sil_scores = {}
        for k in range(2, 11):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(scaled)
            sil_scores[k] = silhouette_score(scaled, labels)

        best_k = max(sil_scores, key=sil_scores.get)

        # ---------------------------------------------------
        # STEP 3: FIT FINAL MODEL
        # ---------------------------------------------------
        final_model = KMeans(n_clusters=best_k, random_state=42)
        merged_df['Cluster'] = final_model.fit_predict(scaled)

        # ---------------------------------------------------
        # STEP 4: USER CHOICES
        # ---------------------------------------------------
        vis_mode = st.radio(
            "Select visualization type:",
            ["Static", "Interactive"],
            horizontal=True
        )

        st.markdown(f"**Optimal number of clusters:** {best_k}")
        st.markdown("Clusters selected automatically using the highest Silhouette score.")

        # ===================================================
        # STATIC VERSION 
        # ===================================================
        if vis_mode == "Static":
            st.sidebar.markdown("### Adjust 3D View")
            elev = st.sidebar.slider("Vertical tilt", 0, 90, 30)
            azim = st.sidebar.slider("Horizontal rotation", 0, 360, 45)
           

            plt.close('all')

            fig = plt.figure(figsize=(5, 4), dpi=150)
            ax = fig.add_subplot(111, projection='3d')

            # PREMIUM COLOR PALETTE
            cmap = plt.cm.coolwarm

            # Scatter plot with nicer styling
            ax.scatter(
                merged_df['Total number of fishing vessels'],
                merged_df['Total Fish Landing (Tonnes)'],
                merged_df['Year'],
                c=merged_df['Cluster'],
                cmap=cmap,
                s=40,
                alpha=0.88,
                edgecolor="white",
                linewidth=0.4,
                depthshade=True
            )

            # =====================================================
            # MODERN GRID & BACKGROUND WITHOUT USING _axinfo
            # =====================================================

            # Light grey background
            ax.set_facecolor("#F5F5F5")

            # Grid style
            ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.3)

            # Tick style
            ax.tick_params(colors="#444", labelsize=7)

            # Label style
            ax.set_xlabel("Vessels", fontsize=8, labelpad=6, color="#333")
            ax.set_ylabel("Landings", fontsize=8, labelpad=6, color="#333")
            ax.set_zlabel("Year", fontsize=8, labelpad=6, color="#333")

            # Title
            ax.set_title(
                f"Static 3D KMeans (k={best_k})",
                fontsize=10,
                weight="bold",
                pad=12,
                color="#222"
            )

            # Camera angles
            ax.view_init(elev=elev, azim=azim)

            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)

        # ===================================================
        # INTERACTIVE VERSION — PLOTLY (FULL 3D ROTATION)
        # ===================================================
        else:
            fig = px.scatter_3d(
                merged_df,
                x='Total number of fishing vessels',
                y='Total Fish Landing (Tonnes)',
                z='Year',
                color='Cluster',
                color_continuous_scale='Viridis',
                symbol='Cluster',
                hover_data=['State', 'Year', 'Total Fish Landing (Tonnes)', 'Total number of fishing vessels'],
                title=f"Interactive 3D KMeans Clustering (k={best_k})",
                height=600
            )

            fig.update_traces(
                marker=dict(
                    size=5,
                    line=dict(width=0.7, color='black')
                )
            )

            fig.update_layout(
                scene=dict(
                    xaxis_title="Vessels",
                    yaxis_title="Landings",
                    zaxis_title="Year",
                    xaxis=dict(backgroundcolor='#1f1f1f'),
                    yaxis=dict(backgroundcolor='#1f1f1f'),
                    zaxis=dict(backgroundcolor='#1f1f1f'),
                ),
                paper_bgcolor='#111111',
                font_color='white',
                margin=dict(l=0, r=0, b=0, t=50)
            )

            st.plotly_chart(fig, use_container_width=True)


    
        

    elif plot_option == "Unified HDBSCAN Outlier Detection":
        import matplotlib.pyplot as plt
        import seaborn as sns
       
        import folium
     
        from streamlit_folium import st_folium

        st.subheader("Unified HDBSCAN Outlier Detection (Monthly + Yearly)")
        st.markdown("<p style='color:#ccc'>Detect both monthly and yearly anomalies with map and explanations.</p>",
                    unsafe_allow_html=True)

        # -----------------------------
        # User selects YEAR
        # -----------------------------
        years = sorted(merged_df["Year"].unique())
        sel_year = st.selectbox("Select Year:", years)

        # =========================================================================
        # 1️⃣ YEARLY OUTLIERS (State-Level Landing vs Vessels)
        # =========================================================================
        st.markdown("## 🟦 Yearly Outliers Summary")

        df_yearly = merged_df[merged_df["Year"] == sel_year].copy()
        df_yearly = df_yearly[[
            "State", "Year",
            "Total Fish Landing (Tonnes)",
            "Total number of fishing vessels"
        ]].dropna()

        df_yearly.rename(columns={
            "Total Fish Landing (Tonnes)": "Landing",
            "Total number of fishing vessels": "Vessels"
        }, inplace=True)

        if df_yearly.empty:
            st.warning("No data available for yearly summary.")
        else:
            # ---------- Scaling ----------
            X_year = StandardScaler().fit_transform(df_yearly[["Landing", "Vessels"]])

            # ---------- HDBSCAN ----------
            yearly_clusterer = hdbscan.HDBSCAN(min_samples=3, min_cluster_size=3,
                                            prediction_data=True).fit(X_year)

            df_yearly["Outlier_Score"] = yearly_clusterer.outlier_scores_
            df_yearly["Outlier_Norm"] = df_yearly["Outlier_Score"] / df_yearly["Outlier_Score"].max()
            df_yearly["Anomaly"] = df_yearly["Outlier_Norm"] >= 0.65

            # ---------- Explanation ----------
            avg_land_y = df_yearly["Landing"].mean()
            avg_ves_y = df_yearly["Vessels"].mean()

            def explain_y(row):
                L, V = row["Landing"], row["Vessels"]
                if L > avg_land_y and V < avg_ves_y:
                    return "⚡ High landing but few vessels → Efficient/Exceptional"
                if L < avg_land_y and V > avg_ves_y:
                    return "🐟 Low catch per vessel → Possible overfishing"
                if L < avg_land_y and V < avg_ves_y:
                    return "🛶 Low activity → Small fleet/Seasonal"
                if L > avg_land_y and V > avg_ves_y:
                    return "⚓ Large operations → Intensive fishing"
                return "Unusual pattern"

            df_yearly["Explanation"] = df_yearly.apply(explain_y, axis=1)

            yearly_outliers = df_yearly[df_yearly["Anomaly"] == True][[
                "State", "Landing", "Vessels", "Outlier_Norm", "Explanation"
            ]]

            if yearly_outliers.empty:
                st.success("No yearly anomalies detected.")
            else:
                st.dataframe(yearly_outliers, use_container_width=True)

            # -------------------- Scatter Plot --------------------
            st.markdown("### 📈 Yearly Landing vs Vessels (Highlighted Outliers)")
            fig, ax = plt.subplots(figsize=(9, 5))

            sns.scatterplot(
                data=df_yearly,
                x="Vessels",
                y="Landing",
                hue="Outlier_Norm",
                palette="viridis",
                s=100,
                ax=ax
            )

            ano = df_yearly[df_yearly["Anomaly"] == True]
            ax.scatter(
                ano["Vessels"], ano["Landing"],
                s=250, facecolors="none",
                edgecolors="red", linewidth=2, label="Outlier"
            )

            for _, r in ano.iterrows():
                ax.text(r["Vessels"] + 0.2, r["Landing"] + 0.2, r["State"],
                        color="red", fontsize=9, fontweight="bold")

            ax.set_title(f"State-Level Outliers ({sel_year})")
            ax.grid(alpha=0.3)
            ax.legend()
            st.pyplot(fig)

            # -------------------- MAP --------------------
            st.markdown("### 🗺️ Map of Yearly Outliers")

            coords = {
                "JOHOR TIMUR/EAST JOHORE": [2.0, 104.1],
                "JOHOR BARAT/WEST JOHORE": [1.9, 103.3],
                "JOHOR": [1.4854, 103.7618],
                "MELAKA": [2.1896, 102.2501],
                "NEGERI SEMBILAN": [2.7258, 101.9424],
                "SELANGOR": [3.0738, 101.5183],
                "PAHANG": [3.8126, 103.3256],
                "TERENGGANU": [5.3302, 103.1408],
                "KELANTAN": [6.1254, 102.2381],
                "PERAK": [4.5921, 101.0901],
                "PULAU PINANG": [5.4164, 100.3327],
                "KEDAH": [6.1184, 100.3685],
                "PERLIS": [6.4449, 100.2048],
                "SABAH": [5.9788, 116.0753],
                "SARAWAK": [1.5533, 110.3592],
                "W.P. LABUAN": [5.2831, 115.2308],
            }

            df_yearly["Coords"] = df_yearly["State"].map(coords)
            m = folium.Map(location=[4.5, 109.5], zoom_start=6)  # ← SAFE NOW

            for _, row in df_yearly.iterrows():
                if row["Coords"] is None:
                    continue
                lat, lon = row["Coords"]
                color = "red" if row["Anomaly"] else "blue"
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=8, color=color, fill=True, fill_color=color,
                    tooltip=row["State"]
                ).add_to(m)

            st_folium(m, height=500, width=800)


        # =========================================================================
        # 2️⃣ MONTHLY OUTLIERS (Detailed anomalies for each month)
        # =========================================================================
        st.markdown("## 🟩 Monthly Outliers Summary")

        df_month = merged_monthly[merged_monthly["Year"] == sel_year].copy()

        all_month_outliers = []

        for month in sorted(df_month["Month"].unique()):
            df_m = df_month[df_month["Month"] == month].copy()
            if len(df_m) < 5:
                continue

            df_m = df_m[[
                "State", "Year", "Month",
                "Fish Landing (Tonnes)",
                "Total number of fishing vessels"
            ]].dropna()

            df_m.rename(columns={
                "Fish Landing (Tonnes)": "Landing",
                "Total number of fishing vessels": "Vessels"
            }, inplace=True)

            # ------- HDBSCAN -------
            X_m = StandardScaler().fit_transform(df_m[["Landing", "Vessels"]])
            cl_m = hdbscan.HDBSCAN(min_samples=3, min_cluster_size=3,
                                prediction_data=True).fit(X_m)

            df_m["Outlier_Score"] = cl_m.outlier_scores_
            if df_m["Outlier_Score"].max() == 0:
                continue

            df_m["Outlier_Norm"] = df_m["Outlier_Score"] / df_m["Outlier_Score"].max()
            df_m["Anomaly"] = df_m["Outlier_Norm"] >= 0.65

            avgL = df_m["Landing"].mean()
            avgV = df_m["Vessels"].mean()

            def explain_m(row):
                L, V = row["Landing"], row["Vessels"]
                if L > avgL and V < avgV:
                    return "⚡ High landing but few vessels"
                if L < avgL and V > avgV:
                    return "🐟 Low catch per vessel"
                if L < avgL and V < avgV:
                    return "🛶 Low activity"
                if L > avgL and V > avgV:
                    return "⚓ Intensive fishing"
                return "Unusual"
            df_m["Explanation"] = df_m.apply(explain_m, axis=1)

            out_m = df_m[df_m["Anomaly"] == True][[
                "Year", "Month", "State",
                "Landing", "Vessels",
                "Outlier_Norm", "Explanation"
            ]]
            if not out_m.empty:
                all_month_outliers.append(out_m)

        if len(all_month_outliers) == 0:
            st.success("No monthly anomalies detected.")
        else:
            final_month = pd.concat(all_month_outliers).sort_values(["Month", "State"])
            st.dataframe(final_month, use_container_width=True)
   
    elif plot_option == "HDBSCAN":
        import matplotlib.pyplot as plt
        import seaborn as sns

        st.subheader("HDBSCAN Outlier Detection — Monthly Anomalies per State")
        st.markdown("<p style='color:#ccc'>Automatically detect abnormal landing–vessel patterns for all months in the selected year.</p>",
                    unsafe_allow_html=True)

        years = sorted(merged_monthly["Year"].unique())
        sel_year = st.selectbox("Select Year:", years)

        df_year = merged_monthly[merged_monthly["Year"] == sel_year].copy()
        if df_year.empty:
            st.error("No data found for this year.")
            st.stop()

        # ----------------------------------------------------
        # Collect outliers for all months
        # ----------------------------------------------------
        all_outliers = []

        for month in sorted(df_year["Month"].unique()):
            df = df_year[df_year["Month"] == month].copy()
            if df.empty or len(df) < 5:
                continue

            required_cols = [
                "State", "Year", "Month",
                "Fish Landing (Tonnes)",
                "Total number of fishing vessels"
            ]
            if any(c not in df.columns for c in required_cols):
                continue

            df = df[required_cols].dropna()

            df.rename(columns={
                "Fish Landing (Tonnes)": "Landing",
                "Total number of fishing vessels": "Vessels"
            }, inplace=True)

            if len(df) < 5:
                continue

            # -----------------------
            # HDBSCAN
            # -----------------------
            X = StandardScaler().fit_transform(df[["Landing", "Vessels"]])

            clusterer = hdbscan.HDBSCAN(
                min_samples=3,
                min_cluster_size=3,
                prediction_data=True
            ).fit(X)

            df["Outlier_Score"] = clusterer.outlier_scores_
            if df["Outlier_Score"].max() == 0:
                continue

            df["Outlier_Norm"] = df["Outlier_Score"] / df["Outlier_Score"].max()
            df["Anomaly"] = df["Outlier_Norm"] >= 0.65

            avg_land = df["Landing"].mean()
            avg_ves = df["Vessels"].mean()

            def explain(row):
                L, V = row["Landing"], row["Vessels"]
                if L > avg_land and V < avg_ves:
                    return "⚡ High landing but few vessels → Efficient catch"
                if L < avg_land and V > avg_ves:
                    return "🐟 Low catch per vessel → Possible overfishing"
                if L < avg_land and V < avg_ves:
                    return "🛶 Low activity → Seasonal or small fleet"
                if L > avg_land and V > avg_ves:
                    return "⚓ Large operations → Intensive fishing"
                return "Unusual pattern"

            df["Explanation"] = df.apply(explain, axis=1)

            outliers_only = df[df["Anomaly"] == True][[
                "Year", "Month", "State",
                "Landing", "Vessels",
                "Outlier_Norm", "Explanation"
            ]]

            if not outliers_only.empty:
                all_outliers.append(outliers_only)

        # ----------------------------
        # Combine
        # ----------------------------
        if len(all_outliers) == 0:
            st.success("No anomalies detected for this year.")
            st.stop()

        final_outliers = pd.concat(all_outliers).sort_values(["Month", "State"])

        # ----------------------------
        # Show table
        # ----------------------------
        st.markdown("### 🔍 Detected State-Level Outliers")
        st.dataframe(final_outliers, use_container_width=True)

  



    elif plot_option == "HDBSCAN Outlier Detection":
        import matplotlib.pyplot as plt
        import seaborn as sns

        st.subheader("HDBSCAN Outlier Detection (State-Level Landing vs Vessels)")
        st.markdown("<p style='color:#ccc'>Detect unusual landing–vessel relationships at the state level.</p>",
                    unsafe_allow_html=True)
      
       

        # --------------------------------------------
        # 1. Select Year
        # --------------------------------------------
        years = sorted(merged_df["Year"].unique())
        sel_year = st.selectbox("Select Year:", years, index=len(years)-1)

        df = merged_df[merged_df["Year"] == sel_year].copy()
        if df.empty:
            st.error("No data for selected year.")
            st.stop()

        # --------------------------------------------
        # 2. Prepare features
        # --------------------------------------------
        df = df[[
            "State",
            "Year",
            "Total Fish Landing (Tonnes)",
            "Total number of fishing vessels"
        ]].dropna()

        df.rename(columns={
            "Total Fish Landing (Tonnes)": "Landing",
            "Total number of fishing vessels": "Vessels"
        }, inplace=True)

        # --------------------------------------------
        # 3. Scaling
        # --------------------------------------------
        X = StandardScaler().fit_transform(df[["Landing", "Vessels"]])

        # --------------------------------------------
        # 4. Run HDBSCAN
        # --------------------------------------------
        clusterer = hdbscan.HDBSCAN(
            min_samples=3,
            min_cluster_size=3,
            prediction_data=True
        ).fit(X)

        df["Cluster"] = clusterer.labels_
        df["Outlier_Score"] = clusterer.outlier_scores_
        df["Outlier_Norm"] = df["Outlier_Score"] / df["Outlier_Score"].max()
        df["Anomaly"] = df["Outlier_Norm"] >= 0.65   # anomaly threshold

        # --------------------------------------------
        # 5. Explanation rules
        # --------------------------------------------
        avg_land = df["Landing"].mean()
        avg_ves = df["Vessels"].mean()

        def explain(row):
            L = row["Landing"]
            V = row["Vessels"]

            if L > avg_land and V < avg_ves:
                return "⚡ High landing but few vessels → Highly efficient or exceptional catch"
            if L < avg_land and V > avg_ves:
                return "🐟 Low catch per vessel → Possible overfishing / low stock"
            if L < avg_land and V < avg_ves:
                return "🛶 Low activity → Small fleet or seasonal downtime"
            if L > avg_land and V > avg_ves:
                return "⚓ Large operations → Unusually intensive fishing scale"
            return "Unusual pattern compared to national averages"

        df["Explanation"] = df.apply(explain, axis=1)

        # --------------------------------------------
        # 6. Outlier Table
        # --------------------------------------------
        st.markdown("### 🔍 Detected State-Level Outliers")

        outliers = df[df["Anomaly"] == True][[
            "State", "Landing", "Vessels", "Outlier_Norm", "Explanation"
        ]]

        if outliers.empty:
            st.success("No significant anomalies detected.")
        else:
            st.dataframe(outliers, use_container_width=True)

        # --------------------------------------------
        # 7. Scatter Plot Visualization (with Aligned Legend)
        # --------------------------------------------

        # 🔹 Create a shared header row so both columns align perfectly
        header_left, header_right = st.columns([3, 1])

        with header_left:
            st.markdown("###  Landing vs Vessels (Highlighted Outliers)")

        with header_right:
            st.markdown("""
            <h4 style="text-align:center; color:white; margin-top:0;">
                How to Read HDBSCAN Membership Colors
            </h4>
            """, unsafe_allow_html=True)


        # 🔹 Now create the real content columns
        col_plot, col_legend = st.columns([3, 1], gap="large")

        with col_plot:

            fig, ax = plt.subplots(figsize=(15, 12))
           

            sns.scatterplot(
                data=df,
                x="Vessels",
                y="Landing",
                hue="Outlier_Norm",
                palette="viridis",
                s=100,
                ax=ax
            )

            # highlight anomalies
            ano = df[df["Anomaly"] == True]
            ax.scatter(
                ano["Vessels"],
                ano["Landing"],
                s=250,
                facecolors="none",
                edgecolors="red",
                linewidth=2,
                label="Outlier"
            )

            # label states
            for _, r in ano.iterrows():
                ax.text(
                    r["Vessels"] + 0.2,
                    r["Landing"] + 0.2,
                    r["State"],
                    color="red",
                    fontsize=9,
                    fontweight="bold"
                )

            ax.set_xlabel("Total Vessels")
            ax.set_ylabel("Total Fish Landing (Tonnes)")
            ax.set_title(f"Outlier Detection ({sel_year})")
            ax.grid(alpha=0.3)
            ax.legend()

            st.pyplot(fig)


        with col_legend:

          
            st.markdown("""
            <div style="
                background-color:#111;
                padding:12px;
                border-radius:12px;
                border-left:none;
                margin-top:0px;
                width:100%;
            ">
                        
           
            <p style='color:#ccc; font-size:14px;'>
                HDBSCAN assigns each point a <b>probability from 0 to 1</b> showing
                confidence in cluster membership (not the cluster number).
            </p>

            
            <table style='color:white; font-size:14px; margin-top:10px;'>
                <tr><td>🟣 <b>0.0</b></td><td>Very weak membership</td></tr>
                <tr><td>🔵 <b>0.2</b></td><td>Weak membership</td></tr>
                <tr><td>🟦 <b>0.4</b></td><td>Medium membership</td></tr>
                <tr><td>🟩 <b>0.6</b></td><td>Strong membership</td></tr>
                <tr><td>🟢 <b>0.8</b></td><td>Very strong membership</td></tr>
                <tr><td>🟡 <b>1.0</b></td><td>Perfect membership</td></tr>
                <tr><td>⭕ <b>Outlier</b></td><td>Explicit anomaly</td></tr>
            </table>

           

            </div>
            """, unsafe_allow_html=True)



        # --------------------------------------------
        # 8. MAP VISUALIZATION
        # --------------------------------------------
        st.markdown("### 🗺️ Map of Anomalous States")

        import folium
        from streamlit_folium import st_folium

        # Coordinates
        coords = {
            "JOHOR TIMUR/EAST JOHORE": [2.0, 104.1],
            "JOHOR BARAT/WEST JOHORE": [1.9, 103.3],
            "JOHOR": [1.4854, 103.7618],
            "MELAKA": [2.1896, 102.2501],
            "NEGERI SEMBILAN": [2.7258, 101.9424],
            "SELANGOR": [3.0738, 101.5183],
            "PAHANG": [3.8126, 103.3256],
            "TERENGGANU": [5.3302, 103.1408],
            "KELANTAN": [6.1254, 102.2381],
            "PERAK": [4.5921, 101.0901],
            "PULAU PINANG": [5.4164, 100.3327],
            "KEDAH": [6.1184, 100.3685],
            "PERLIS": [6.4449, 100.2048],
            "SABAH": [5.9788, 116.0753],
            "SARAWAK": [1.5533, 110.3592],
            "W.P. LABUAN": [5.2831, 115.2308],
        }

        df["Coords"] = df["State"].map(coords)

        m = folium.Map(location=[4.5, 109.5], zoom_start=6)

        for _, row in df.iterrows():
            if row["Coords"] is None:
                continue

            lat, lon = row["Coords"]
            color = "red" if row["Anomaly"] else "blue"

            folium.CircleMarker(
                location=[lat, lon],
                radius=8,
                color=color,
                fill=True,
                fill_color=color,
                tooltip=row["State"],
                popup=(
                    f"<b>{row['State']}</b><br>"
                    f"Landing: {row['Landing']:.0f} tonnes<br>"
                    f"Vessels: {row['Vessels']:.0f}<br>"
                    f"Score: {row['Outlier_Norm']:.2f}<br>"
                    f"<i>{row['Explanation']}</i>"
                ),
            ).add_to(m)

        st_folium(m, height=550, width=800)





    elif plot_option == "Automatic DBSCAN":
        import numpy as np  
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy.spatial import ConvexHull

        st.subheader("Automatic DBSCAN Clustering & Outlier Detection")
     
        
        # -----------------------------
        # 1. FILTER VALID STATES
        # -----------------------------
        valid_states = [
            "JOHOR TIMUR/EAST JOHORE", "JOHOR BARAT/WEST JOHORE", "JOHOR",
            "MELAKA", "NEGERI SEMBILAN", "SELANGOR", "PAHANG", "TERENGGANU",
            "KELANTAN", "PERAK", "PULAU PINANG", "KEDAH", "PERLIS",
            "SABAH", "SARAWAK", "W.P. LABUAN"
        ]
        merged_df = merged_df[merged_df["State"].isin(valid_states)].reset_index(drop=True)

        if merged_df.empty:
            st.warning("No valid data after filtering states.")
            st.stop()

        # -----------------------------
        # 2. PREPARE FEATURES
        # -----------------------------
        features = merged_df[["Total Fish Landing (Tonnes)", "Total number of fishing vessels"]]
        scaled = StandardScaler().fit_transform(features)

        n_samples = scaled.shape[0]         # <– FIXED
        n_features = scaled.shape[1]

        # -----------------------------
        # 3. AUTO min_samples
        # -----------------------------
        min_samples_auto = max(3, int(np.log(n_samples)) + n_features)

        # -----------------------------
        # 4. K-distance graph
        # -----------------------------
        neigh = NearestNeighbors(n_neighbors=min_samples_auto)
        distances, _ = neigh.fit(scaled).kneighbors(scaled)
        distances = np.sort(distances[:, min_samples_auto - 1])

        kneedle = KneeLocator(range(len(distances)), distances, curve="convex", direction="increasing")
        eps_auto = distances[kneedle.knee] if kneedle.knee else np.percentile(distances, 90)

        st.markdown(f"**Automatically estimated ε (epsilon):** `{eps_auto:.3f}`")
        st.markdown(f"**Automatically selected min_samples:** `{min_samples_auto}`")

        # -----------------------------
        # 5. K-distance PLOT
        # -----------------------------
        fig_k, ax_k = plt.subplots(figsize=(8, 3.5))
        ax_k.plot(distances)
        if kneedle.knee:
            ax_k.axvline(kneedle.knee, color="red", linestyle="--")
            ax_k.axhline(eps_auto, color="green", linestyle="--")
        ax_k.set_title("K-distance Graph (Auto ε Detection)")
        ax_k.set_xlabel("Sorted points")
        ax_k.set_ylabel("Distance")
        st.pyplot(fig_k)

        # -----------------------------
        # 6. RUN DBSCAN
        # -----------------------------
        db = DBSCAN(eps=eps_auto, min_samples=min_samples_auto)
        labels = db.fit_predict(scaled)
        merged_df["DBSCAN_Label"] = labels

        # -----------------------------
        # 7. SILHOUETTE SCORE
        # -----------------------------
        unique_labels = set(labels) - {-1}
        if len(unique_labels) > 1:
            sil = silhouette_score(scaled[labels != -1], labels[labels != -1])
            st.info(f"Silhouette Score (clusters only): `{sil:.3f}`")
        else:
            st.warning("Silhouette unavailable — only one cluster or all noise.")

        # -----------------------------
        # 8. CLUSTER VISUALIZATION
        # -----------------------------
        fig, ax = plt.subplots(figsize=(10, 6))
        palette = sns.color_palette("bright", len(unique_labels) + 1)

        for label in np.unique(labels):
            pts = scaled[labels == label]

            if label == -1:
                ax.scatter(pts[:, 1], pts[:, 0], s=50, c="lightgray", edgecolor="k",
                        alpha=0.6, label="Noise")
            else:
                color = palette[label % len(palette)]
                ax.scatter(pts[:, 1], pts[:, 0], s=60, c=[color], edgecolor="k",
                        alpha=0.85, label=f"Cluster {label} ({len(pts)})")

                # Convex Hull
                if len(pts) >= 3:
                    hull = ConvexHull(pts)
                    hv = list(hull.vertices) + [hull.vertices[0]]
                    ax.plot(pts[hv, 1], pts[hv, 0], color=color, linewidth=2)

        ax.set_title(f"DBSCAN (eps={eps_auto:.3f}, min_samples={min_samples_auto})")
        ax.set_xlabel("Vessels (scaled)")
        ax.set_ylabel("Landings (scaled)")
        ax.grid(alpha=0.3)
        ax.legend()
        st.pyplot(fig)

        # -----------------------------
        # 9. CLUSTER SUMMARY
        # -----------------------------
        cluster_summary = merged_df[labels != -1].groupby("DBSCAN_Label")[
            ["Total Fish Landing (Tonnes)", "Total number of fishing vessels"]
        ].mean().reset_index()

        st.markdown("### 📊 Cluster Summary")
        st.dataframe(cluster_summary)

        # -----------------------------
        # 10. OUTLIER ANALYSIS
        # -----------------------------
        outliers = merged_df[labels == -1]
        n_outliers = len(outliers)
        st.success(f"Detected {n_outliers} outliers.")

        if n_outliers > 0:
            avg_land = merged_df["Total Fish Landing (Tonnes)"].mean()
            avg_ves = merged_df["Total number of fishing vessels"].mean()

            def explain(r):
                if r["Total Fish Landing (Tonnes)"] > avg_land and r["Total number of fishing vessels"] < avg_ves:
                    return "⚠️ High landing but low vessels – anomaly"
                if r["Total Fish Landing (Tonnes)"] < avg_land and r["Total number of fishing vessels"] > avg_ves:
                    return "🐟 Low catch per vessel – possible overfishing"
                if r["Total Fish Landing (Tonnes)"] < avg_land and r["Total number of fishing vessels"] < avg_ves:
                    return "🛶 Low activity – Possible  seasonal or small fleet"
                return "Atypical pattern vs national average"

            outliers["Why Flagged"] = outliers.apply(explain, axis=1)
            st.markdown("### 🚨 Outlier Details")
            st.dataframe(outliers)

            # Heatmap
            fig_h, ax_h = plt.subplots(figsize=(8, 4))
            sns.heatmap(outliers[
                ["Total Fish Landing (Tonnes)", "Total number of fishing vessels"]
            ], annot=True, fmt=".0f", cmap="coolwarm", cbar=False, ax=ax_h)
            ax_h.set_title("Outlier Heatmap")
            st.pyplot(fig_h)


    

       

                    
    elif plot_option == "Hierarchical Clustering":
                        
            st.subheader("Hierarchical Clustering of Malaysian States")
            
                # Call the hierarchical clustering function
            hierarchical_clustering(merged_df)
            
    elif plot_option == "Geospatial Map":
                st.subheader("Geospatial Distribution of Fish Landings by Year and Region")

            # Let user choose year
                available_years = sorted(merged_df['Year'].unique())
                selected_year = st.selectbox("Select Year", available_years, index=len(available_years)-1)

            # Filter dataset by selected year
                geo_df = merged_df[merged_df['Year'] == selected_year].copy()

                import re
                import folium
                from streamlit_folium import st_folium

            # Manually define coordinates for each region (including subregions)
                state_coords = {
                # Johor regions
                    "JOHOR TIMUR/EAST JOHORE": [2.0, 104.1],
                    "JOHOR BARAT/WEST JOHORE": [1.9, 103.3],
                    "JOHOR": [1.4854, 103.7618],
                    "MELAKA": [2.1896, 102.2501],
                    "NEGERI SEMBILAN": [2.7258, 101.9424],
                    "SELANGOR": [3.0738, 101.5183],
                    "PAHANG": [3.8126, 103.3256],
                    "TERENGGANU": [5.3302, 103.1408],
                    "KELANTAN": [6.1254, 102.2381],
                    "PERAK": [4.5921, 101.0901],
                    "PULAU PINANG": [5.4164, 100.3327],
                    "KEDAH": [6.1184, 100.3685],
                    "PERLIS": [6.4449, 100.2048],
                    "SABAH": [5.9788, 116.0753],
                    "SARAWAK": [1.5533, 110.3592],
                    "W.P. LABUAN": [5.2831, 115.2308]
            }

                # Clean state names in dataset (remove spaces and unify slashes)
                geo_df['State_Clean'] = (
                    geo_df['State']
                    .astype(str)
                    .str.upper()
                    .str.replace(r'\s*/\s*', '/', regex=True)  # Normalize " / " to "/"
                    .str.replace(r'\s+', ' ', regex=True)      # Remove multiple spaces
                    .str.strip()
                )

            # Clean coordinate dictionary
                clean_coords = { re.sub(r'\s*/\s*', '/', k.upper().strip()): v for k, v in state_coords.items() }

                
                # Clean coordinate dictionary keys the same way
            
        # Now safely map using the cleaned version
                geo_df['Coords'] = geo_df['State_Clean'].map(clean_coords)

            # Drop regions with no coordinates (to avoid map crash)
                missing_coords = geo_df[geo_df['Coords'].isna()]['State'].unique()
                if len(missing_coords) > 0:
                    st.warning(f"No coordinates found for: {', '.join(missing_coords)}")

                geo_df = geo_df.dropna(subset=['Coords'])

                #  Safety check: make sure there’s data to map
                if geo_df.empty:
                    st.warning("No valid locations found for the selected year.")
                else:
                # Create Folium map centered on Malaysia
                    m = folium.Map(location=[4.5, 109.5], zoom_start=6)

        

            # Add markers for each region
                    for _, row in geo_df.iterrows():
                        folium.CircleMarker(
                            location=row['Coords'],
                            radius=8,
                            color='blue',
                            fill=True,
                            fill_color='cyan',
                            popup=f"<b>{row['State']}</b><br>"
                                f"Fish Landing: {row['Total Fish Landing (Tonnes)']:.2f} tonnes<br>"
                                f"Vessels: {row['Total number of fishing vessels']:.0f}",
                            tooltip=row['State']
                        ).add_to(m)

            # Display map
                    st_folium(m, width=800, height=500)

    
    elif plot_option == "Interactive Geospatial Map":
            st.subheader("Geospatial Distribution of Fish Landings by Year and Region")
        
            import re
            import folium
            import branca.colormap as cm
            from folium.plugins import MarkerCluster, MiniMap, Fullscreen, HeatMap
            from streamlit_folium import st_folium
            from streamlit_js_eval import streamlit_js_eval

            valid_states = ["JOHOR", "JOHOR BARAT/WEST JOHORE", "JOHOR TIMUR/EAST JOHORE",
                "MELAKA", "NEGERI SEMBILAN", "SELANGOR", "PAHANG", "TERENGGANU",
                "KELANTAN", "PERAK", "PULAU PINANG", "KEDAH", "PERLIS",
                "SABAH", "SARAWAK", "W.P. LABUAN"
            ]
            
            merged_df = merged_df[merged_df['State'].isin(valid_states)]
            # --- Step 1: User Filters ---
            available_years = sorted(merged_df['Year'].unique())
            selected_year = st.selectbox("Select Year", available_years, index=len(available_years) - 1)
        
            available_states = sorted(merged_df['State'].unique())
            selected_states = st.multiselect("Select State(s)",options=available_states,default=available_states,placeholder="Choose one or more states to display",label_visibility="visible")                   
        
            # Filter dataset
            geo_df = merged_df[
                (merged_df['Year'] == selected_year) &
                (merged_df['State'].isin(selected_states))
            ].copy()
        
            # --- Step 2: Define Coordinates ---
            state_coords = {
                "JOHOR TIMUR/EAST JOHORE": [2.0, 104.1],
                "JOHOR BARAT/WEST JOHORE": [1.9, 103.3],
                "JOHOR": [1.4854, 103.7618],
                "MELAKA": [2.1896, 102.2501],
                "NEGERI SEMBILAN": [2.7258, 101.9424],
                "SELANGOR": [3.0738, 101.5183],
                "PAHANG": [3.8126, 103.3256],
                "TERENGGANU": [5.3302, 103.1408],
                "KELANTAN": [6.1254, 102.2381],
                "PERAK": [4.5921, 101.0901],
                "PULAU PINANG": [5.4164, 100.3327],
                "KEDAH": [6.1184, 100.3685],
                "PERLIS": [6.4449, 100.2048],
                "SABAH": [5.9788, 116.0753],
                "SARAWAK": [1.5533, 110.3592],
                "W.P. LABUAN": [5.2831, 115.2308]
            }
        
            # --- Step 3: Clean Names & Map Coordinates ---
            geo_df['State_Clean'] = (
                geo_df['State']
                .astype(str)
                .str.upper()
                .str.replace(r'\s*/\s*', '/', regex=True)
                .str.replace(r'\s+', ' ', regex=True)
                .str.strip()
            )
        
            clean_coords = {
                re.sub(r'\s*/\s*', '/', k.upper().strip()): v for k, v in state_coords.items()
            }
        
            geo_df['Coords'] = geo_df['State_Clean'].map(clean_coords)
        
            # --- Step 4: Handle Missing Data ---
            missing_coords = geo_df[geo_df['Coords'].isna()]['State'].unique()
            if len(missing_coords) > 0:
                st.warning(f"No coordinates found for: {', '.join(missing_coords)}")
        
            geo_df = geo_df.dropna(subset=['Coords'])
            if geo_df.empty:
                st.warning("No valid locations found for the selected year.")
        
                # --- Step 5: Create Base Map ---
            # m = folium.Map(location=[4.5, 109.5], zoom_start=6, tiles="CartoDB positron")
# --- Step 5: Create Base Map ---
# Compute automatic bounds to include all states tightly (Peninsular + Borneo)
           # --- Step 5: Create Base Map (with Theme Toggle) ---
            # --- Map Theme Selection (top-left area above map) ---
            st.markdown("### Map Theme")
            map_theme = st.radio(
                "Choose Map Theme:",
                ["Light Mode", "Dark Mode", "Satellite", "Default"],
                horizontal=True,
                key="map_theme_radio"
            )
            
            # Apply tile according to theme
            tile_map = {
                "Light Mode": "CartoDB positron",
                "Dark Mode": "CartoDB dark_matter",
                "Satellite": "Esri.WorldImagery",
                "Default": "OpenStreetMap"
            }
            
        # --- Step 5: Create Base Map ---
            lat_min = geo_df["Coords"].apply(lambda x: x[0]).min()
            lat_max = geo_df["Coords"].apply(lambda x: x[0]).max()
            lon_min = geo_df["Coords"].apply(lambda x: x[1]).min()
            lon_max = geo_df["Coords"].apply(lambda x: x[1]).max()
        
            m = folium.Map(location=[4.2, 108.0], zoom_start=6.7, tiles=None)
                    # Apply selected tile theme (with clean label)
            folium.TileLayer(
                tiles=tile_map[map_theme],
                name="Base Map",          # clean name for control
                attr="© OpenStreetMap & Esri contributors", 
                control=False             # hide from layer list
            ).add_to(m)
            m.fit_bounds([[lat_min, lon_min], [lat_max, lon_max]], padding=(10, 10))
             
            # --- Step 6: Add Color Scale ---
            min_val = float(geo_df['Total Fish Landing (Tonnes)'].min())
            max_val = float(geo_df['Total Fish Landing (Tonnes)'].max())
            
            colormap = cm.LinearColormap(
                colors=['blue', 'lime', 'yellow', 'orange', 'red'],
                vmin=min_val,
                vmax=max_val,
                caption=f"Fish Landing (Tonnes)\nMin: {min_val:,.0f}  |  Max: {max_val:,.0f}"
            )
            colormap.add_to(m)
            
            # --- Step 7: Add Circle Markers ---
            for _, row in geo_df.iterrows():
                popup_html = (
                    f"<b>{row['State']}</b><br>"
                    f"Fish Landing: {row['Total Fish Landing (Tonnes)']:.2f} tonnes<br>"
                    f"Fish Vessels: {row['Total number of fishing vessels']:.0f}"
                )
                color = colormap(row['Total Fish Landing (Tonnes)'])
                folium.CircleMarker(
                    location=row['Coords'],
                    radius=9,
                    color="black",
                    weight=1,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.85,
                    popup=folium.Popup(popup_html, max_width=250),
                    tooltip=row["State"],
                ).add_to(m)
            
            # --- Step 8: Add Heatmap Layer ---
            geo_df['HeatValue'] = np.log1p(geo_df['Total Fish Landing (Tonnes)'])
            heat_data = [
                [row['Coords'][0], row['Coords'][1], row['HeatValue']]
                for _, row in geo_df.iterrows()
            ]
            gradient = {
                0.0: 'blue',
                0.3: 'lime',
                0.5: 'yellow',
                0.7: 'orange',
                1.0: 'red'
            }
            HeatMap(
                heat_data,
                name="Fish Landing Heatmap",
                radius=15,
                blur=8,
                min_opacity=0.5,
                max_opacity=0.95,
                gradient=gradient,
                max_val=geo_df["Total Fish Landing (Tonnes)"].max(),
            ).add_to(m)
            
            # --- Step 9: Map Controls ---
            MiniMap(toggle_display=True, zoom_level_fixed=6).add_to(m)
            Fullscreen(position='topright').add_to(m)
            folium.LayerControl(collapsed=False).add_to(m)
            
            # --- Step 10: Display Map ---
            st_folium(m, use_container_width=True, height=600)
            
            # --- Step 11: Summary Section ---
            st.markdown(f"""
            **Summary for {selected_year}:**
            - 🟢 States displayed: {len(selected_states)}
            - ⚓ Total fish landing: {geo_df['Total Fish Landing (Tonnes)'].sum():,.0f} tonnes
            - 🚢 Total vessels: {geo_df['Total number of fishing vessels'].sum():,}
            """)
            
            with st.expander("ℹ️ Color Legend for Fish Landing Intensity", expanded=True):
                st.markdown("""
                **Color Interpretation:**
                - 🟥 **Red / Orange** → High fish landing states  
                - 🟨 **Yellow / Lime** → Medium fish landing  
                - 🟦 **Blue / Green** → Low fish landing  
                <br>
                The heatmap shows **relative fish landing intensity by region**.
                """, unsafe_allow_html=True)


    


    elif plot_option == "Geospatial Map(Heatmap)":
        import folium
        from streamlit_folium import st_folium
        from folium.plugins import HeatMap
        from branca.colormap import linear

        st.subheader("🌍 Interactive Geospatial Heatmap")
        st.markdown("""
        <p style='color:#ccc'>
        Explore Malaysia’s fish landing distribution using an intuitive interactive heatmap.
        </p>
        """, unsafe_allow_html=True)

        # ----------------------------------------------------
        # CREATE UI CONTAINERS FOR LAYOUT ORDER
        # ----------------------------------------------------
        summary_container = st.container()
        selection_container = st.container()
        map_container = st.container()
        table_container = st.container()
        interpretation_container = st.container()

        # ----------------------------------------------------
        # 1️⃣ YEAR SELECTION (but shown AFTER summary via container)
        # ----------------------------------------------------
        with selection_container:
            years = sorted(merged_df["Year"].unique())
            sel_year = st.selectbox("Select Year:", years, index=len(years)-1)

        # ----------------------------------------------------
        # PROCESS YEARLY DATA
        # ----------------------------------------------------
        df_year = merged_df[merged_df["Year"] == sel_year].copy()
        df_year = df_year.groupby("State", as_index=False)[
            ["Total Fish Landing (Tonnes)", "Total number of fishing vessels"]
        ].sum()

        df_year.rename(columns={
            "Total Fish Landing (Tonnes)": "Landing",
            "Total number of fishing vessels": "Vessels"
        }, inplace=True)

        df_year = df_year[df_year["Landing"] > 0]

        # ----------------------------------------------------
        # STATE COORDINATES
        # ----------------------------------------------------
        coords = {
            "JOHOR TIMUR/EAST JOHORE": [2.0, 104.1],
            "JOHOR BARAT/WEST JOHORE": [1.9, 103.3],
            "JOHOR": [1.4854, 103.7618],
            "MELAKA": [2.1896, 102.2501],
            "NEGERI SEMBILAN": [2.7258, 101.9424],
            "SELANGOR": [3.0738, 101.5183],
            "PAHANG": [3.8126, 103.3256],
            "TERENGGANU": [5.3302, 103.1408],
            "KELANTAN": [6.1254, 102.2381],
            "PERAK": [4.5921, 101.0901],
            "PULAU PINANG": [5.4164, 100.3327],
            "KEDAH": [6.1184, 100.3685],
            "PERLIS": [6.4449, 100.2048],
            "SABAH": [5.9788, 116.0753],
            "SARAWAK": [1.5533, 110.3592],
            "W.P. LABUAN": [5.2831, 115.2308],
        }

        df_year["Coords"] = df_year["State"].map(coords)
        df_year = df_year.dropna(subset=["Coords"]).copy()

        # ----------------------------------------------------
        # 2️⃣ SUMMARY CARDS (TOP SCREEN)
        # ----------------------------------------------------
        with summary_container:
            total = df_year["Landing"].sum()
            highest = df_year.loc[df_year["Landing"].idxmax()]
            lowest = df_year.loc[df_year["Landing"].idxmin()]

            card = """
                background:#1e1e1e; padding:15px;
                border-radius:10px; border:1px solid #333;
            """

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div style="{card}">
                    <div style="color:#ccc">Total Landing</div>
                    <div style="color:white;font-size:26px;"><b>{total:,.0f}</b> tonnes</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div style="{card}">
                    <div style="color:#ccc">Highest State</div>
                    <div style="color:#4ade80;font-size:18px;"><b>{highest['State']}</b></div>
                    <div style="color:white;font-size:26px;"><b>{highest['Landing']:,.0f}</b> t</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div style="{card}">
                    <div style="color:#ccc">Lowest State</div>
                    <div style="color:#f87171;font-size:18px;"><b>{lowest['State']}</b></div>
                    <div style="color:white;font-size:26px;"><b>{lowest['Landing']:,.0f}</b> t</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # ----------------------------------------------------
        # 3️⃣ STATE MULTISELECT (AFTER YEAR SELECTOR)
        # ----------------------------------------------------
        with selection_container:
            all_states = sorted(df_year["State"].unique())
            selected_states = st.multiselect(
                "Select State(s):",
                all_states,
                default=all_states
            )

        df = df_year[df_year["State"].isin(selected_states)].copy()

        if df.empty:
            st.warning("No states selected.")
            st.stop()

        # ----------------------------------------------------
        # 4️⃣ MAP THEME SELECTOR
        # ----------------------------------------------------
        with selection_container:
            theme = st.radio(
                "Choose Map Theme:",
                ["Light", "Dark", "Satellite", "Default"],
                horizontal=True
            )

        tile_map = {
            "Light": "CartoDB positron",
            "Dark": "CartoDB dark_matter",
            "Satellite": "Esri.WorldImagery",
            "Default": "OpenStreetMap"
        }

        # ----------------------------------------------------
        # 5️⃣ MAP (Heatmap + Markers)
        # ----------------------------------------------------
        min_lat = min(df["Coords"].apply(lambda x: x[0]))
        max_lat = max(df["Coords"].apply(lambda x: x[0]))
        min_lon = min(df["Coords"].apply(lambda x: x[1]))
        max_lon = max(df["Coords"].apply(lambda x: x[1]))

        m = folium.Map(tiles=None, zoom_start=6)
        folium.TileLayer(tile_map[theme], name="Base Map", control=False).add_to(m)
        m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

        # Legend
        min_v = df["Landing"].min()
        max_v = df["Landing"].max()

        cmap = linear.Blues_09.scale(min_v, max_v).to_step(5)
        cmap.caption = f"Fish Landing (Tonnes)\nMin: {min_v:,.0f} | Max: {max_v:,.0f}"
        m.add_child(cmap)

        # HEATMAP
        heat_group = folium.FeatureGroup("Heatmap")
        heat_data = [[r["Coords"][0], r["Coords"][1], r["Landing"]] for _, r in df.iterrows()]
        HeatMap(heat_data, radius=40, blur=25, min_opacity=0.4).add_to(heat_group)
        heat_group.add_to(m)

        # MARKERS
        marker_group = folium.FeatureGroup("State Markers")
        for _, r in df.iterrows():
            folium.CircleMarker(
                location=r["Coords"],
                radius=9,
                color="white",
                fill=True,
                fill_color=cmap(r["Landing"]),
                fill_opacity=0.95,
                weight=1.3,
                tooltip=f"<b>{r['State']}</b><br>{r['Landing']:,.0f} tonnes"
            ).add_to(marker_group)

        marker_group.add_to(m)
        folium.LayerControl().add_to(m)

        with map_container:
            st_folium(m, height=550, width="100%")

        # ----------------------------------------------------
        # 6️⃣ TABLE
        # ----------------------------------------------------
        with table_container:
            st.markdown("### 📋 State Landing Table")
            st.dataframe(
                df.sort_values("Landing", ascending=False).reset_index(drop=True),
                use_container_width=True,
                height=300
            )

        # ----------------------------------------------------
        # 7️⃣ INTERPRETATION
        # ----------------------------------------------------
        with interpretation_container:
            with st.expander("ℹ️ How to read this map"):
                st.markdown("""
                **Heatmap intensity** reflects total fish landing:
                - Darker blue → Higher landing  
                - Light blue → Lower landing  
                - Hover markers to see exact values  
                """)
    elif plot_option == "Geospatial Map (Upgraded)":
        import folium
        import numpy as np
        from folium.plugins import HeatMap, MiniMap, Fullscreen
        from streamlit_folium import st_folium
        from branca.colormap import linear

        st.subheader("🌍 Upgraded Geospatial Heatmap (Landing + Vessels + Efficiency)")
        st.markdown("""
        <p style='color:#ccc'>
        This upgraded geospatial map shows:
        <br>• Fish Landing Heatmap
        <br>• Vessel Count Heatmap
        <br>• Efficiency Heatmap (Landing ÷ Vessel)
        <br>• State markers with detailed popup info
        <br>• Map theme selector and layer control
        </p>
        """, unsafe_allow_html=True)

        # -------------------------
        # CONTAINERS
        # -------------------------
        summary_c = st.container()
        selection_c = st.container()
        map_c = st.container()
        table_c = st.container()
        info_c = st.container()

        # -------------------------
        # YEAR SELECTION
        # -------------------------
        with selection_c:
            years = sorted(merged_df["Year"].unique())
            sel_year = st.selectbox("Select Year:", years, index=len(years)-1)

        df_year = merged_df[merged_df["Year"] == sel_year].copy()

        df_year = df_year.groupby("State", as_index=False)[
            ["Total Fish Landing (Tonnes)", "Total number of fishing vessels"]
        ].sum()

        df_year.rename(columns={
            "Total Fish Landing (Tonnes)": "Landing",
            "Total number of fishing vessels": "Vessels"
        }, inplace=True)

        df_year = df_year[df_year["Landing"] > 0].copy()

        # -------------------------
        # DEFINE COORDINATES ONCE
        # -------------------------
        coordinates = {
            "JOHOR TIMUR/EAST JOHORE": [2.0, 104.1],
            "JOHOR BARAT/WEST JOHORE": [1.9, 103.3],
            "JOHOR": [1.4854, 103.7618],
            "MELAKA": [2.1896, 102.2501],
            "NEGERI SEMBILAN": [2.7258, 101.9424],
            "SELANGOR": [3.0738, 101.5183],
            "PAHANG": [3.8126, 103.3256],
            "TERENGGANU": [5.3302, 103.1408],
            "KELANTAN": [6.1254, 102.2381],
            "PERAK": [4.5921, 101.0901],
            "PULAU PINANG": [5.4164, 100.3327],
            "KEDAH": [6.1184, 100.3685],
            "PERLIS": [6.4449, 100.2048],
            "SABAH": [5.9788, 116.0753],
            "SARAWAK": [1.5533, 110.3592],
            "W.P. LABUAN": [5.2831, 115.2308],
        }

        df_year["Coords"] = df_year["State"].map(coordinates)
        df_year = df_year.dropna(subset=["Coords"]).copy()

        df_year["Efficiency"] = df_year["Landing"] / df_year["Vessels"]

        # -------------------------
        # SUMMARY CARDS
        # -------------------------
        with summary_c:
            total_land = df_year["Landing"].sum()
            total_vess = df_year["Vessels"].sum()
            high = df_year.loc[df_year["Landing"].idxmax()]
            low = df_year.loc[df_year["Landing"].idxmin()]

            style_card = """
                background: #1e1e1e;
                padding: 15px;
                border-radius: 10px;
                border: 1px solid #333;
            """

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(f"""
                <div style="{style_card}">
                    <div style="color:#ccc">Total Landing</div>
                    <div style="color:white; font-size:26px;"><b>{total_land:,.0f}</b></div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div style="{style_card}">
                    <div style="color:#ccc">Total Vessels</div>
                    <div style="color:white; font-size:26px;"><b>{total_vess:,.0f}</b></div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div style="{style_card}">
                    <div style="color:#ccc">Highest Landing</div>
                    <div style="color:#4ade80;font-size:18px;"><b>{high['State']}</b></div>
                    <div style="color:white;font-size:26px;"><b>{high['Landing']:,.0f}</b></div>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                st.markdown(f"""
                <div style="{style_card}">
                    <div style="color:#ccc">Lowest Landing</div>
                    <div style="color:#f87171;font-size:18px;"><b>{low['State']}</b></div>
                    <div style="color:white;font-size:26px;"><b>{low['Landing']:,.0f}</b></div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # -------------------------
        # STATE FILTER
        # -------------------------
        with selection_c:
            state_list = sorted(df_year["State"].unique())
            selected_states = st.multiselect("Select State(s):", state_list, default=state_list)

        df = df_year[df_year["State"].isin(selected_states)].copy()

        if df.empty:
            st.warning("No states selected.")
            st.stop()

        # -------------------------
        # MAP THEME
        # -------------------------
        with selection_c:
            map_theme = st.radio("Choose Map Theme:", ["Light", "Dark", "Satellite", "Default"], horizontal=True)

        tile_map = {
            "Light": "CartoDB positron",
            "Dark": "CartoDB dark_matter",
            "Satellite": "Esri.WorldImagery",
            "Default": "OpenStreetMap"
        }

        # -------------------------
        # CREATE MAP
        # -------------------------
        m = folium.Map(location=[4.2, 108.5], zoom_start=6, tiles=None)
        folium.TileLayer(tile_map[map_theme], name="Base Map", control=False).add_to(m)

        lat_min = df["Coords"].apply(lambda x: x[0]).min()
        lat_max = df["Coords"].apply(lambda x: x[0]).max()
        lon_min = df["Coords"].apply(lambda x: x[1]).min()
        lon_max = df["Coords"].apply(lambda x: x[1]).max()

        m.fit_bounds([[lat_min, lon_min], [lat_max, lon_max]])

        # ---------------------------------------
        # COLOR SCALES
        # ---------------------------------------
        land_cmap = linear.Blues_09.scale(df["Landing"].min(), df["Landing"].max())
        ves_cmap = linear.YlOrRd_09.scale(df["Vessels"].min(), df["Vessels"].max())
        eff_cmap = linear.PuRd_09.scale(df["Efficiency"].min(), df["Efficiency"].max())

        # ---------------------------------------
        # HEATMAPS (3 Layers)
        # ---------------------------------------
        layer_land = folium.FeatureGroup("Landing Heatmap")
        heat_land = [[c[0], c[1], v] for c, v in zip(df["Coords"], df["Landing"])]
        HeatMap(heat_land, radius=40, blur=25, min_opacity=0.4).add_to(layer_land)
        layer_land.add_to(m)

        layer_vess = folium.FeatureGroup("Vessels Heatmap")
        heat_vess = [[c[0], c[1], v] for c, v in zip(df["Coords"], df["Vessels"])]
        HeatMap(heat_vess, radius=40, blur=25,
                gradient={0.2:"blue", 0.5:"cyan", 0.7:"lime", 1:"red"}
        ).add_to(layer_vess)
        layer_vess.add_to(m)

        layer_eff = folium.FeatureGroup("Efficiency Heatmap")
        heat_eff = [[c[0], c[1], v] for c, v in zip(df["Coords"], df["Efficiency"])]
        HeatMap(heat_eff, radius=40, blur=30,
                gradient={0.2:"purple", 0.5:"magenta", 0.8:"pink", 1:"white"}
        ).add_to(layer_eff)
        layer_eff.add_to(m)

        # ---------------------------------------
        # MARKERS
        # ---------------------------------------
        marker_layer = folium.FeatureGroup("State Markers")

        for _, r in df.iterrows():
            lat, lon = r["Coords"]
            popup = (
                f"<b>{r['State']}</b><br>"
                f"Landing: {r['Landing']:,.0f} t<br>"
                f"Vessels: {r['Vessels']:,.0f}<br>"
                f"Efficiency: {r['Efficiency']:.2f}"
            )

            folium.CircleMarker(
                location=[lat, lon],
                radius=9,
                color="black",
                weight=1,
                fill=True,
                fill_color=land_cmap(r["Landing"]),
                fill_opacity=0.9,
                popup=folium.Popup(popup, max_width=250),
                tooltip=r["State"]
            ).add_to(marker_layer)

        marker_layer.add_to(m)

        # MiniMap & Fullscreen
        MiniMap(toggle_display=True, zoom_level_fixed=6).add_to(m)
        Fullscreen(position='topright').add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)

        # ---------------------------------------
        # DISPLAY MAP
        # ---------------------------------------
        with map_c:
            st_folium(m, height=600, width="100%")

        # ---------------------------------------
        # TABLE
        # ---------------------------------------
        with table_c:
            st.markdown("### 📊 State Landing / Vessels / Efficiency")
            st.dataframe(
                df.sort_values("Landing", ascending=False).reset_index(drop=True),
                use_container_width=True,
                height=350
            )

        # ---------------------------------------
        # INTERPRETATION
        # ---------------------------------------
        with info_c:
            with st.expander("ℹ️ How to interpret the map"):
                st.markdown("""
                ### Layers Explained:
                - **Landing Heatmap (Blue)** – total fish landing  
                - **Vessels Heatmap (Red gradient)** – number of fishing vessels  
                - **Efficiency Heatmap (Purple → White)** – landing per vessel  

                ### Marker Colors:
                - Marker color indicates **landing amount**
                - Hover to view landing, vessels, efficiency  
                """)


    elif plot_option == "Geospatial Map (Upgraded)2":
        import folium
        import numpy as np
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        from folium.plugins import HeatMap, MiniMap, Fullscreen
        from streamlit_folium import st_folium
        from branca.colormap import linear

        st.subheader("🌍 Geospatial Heatmap ")
       

        summary_c = st.container()
        selection_c = st.container()
        map_c = st.container()
        table_c = st.container()
        chart_c = st.container()
        info_c = st.container()

        # -------------------------------------------------------------------
        # YEAR SELECTION
        # -------------------------------------------------------------------
        with selection_c:
            years = sorted(merged_df["Year"].unique())
            sel_year = st.selectbox("Select Year:", years, index=len(years)-1)

        df_year = merged_df[merged_df["Year"] == sel_year].copy()

        df_year = df_year.groupby("State", as_index=False)[
            ["Total Fish Landing (Tonnes)", "Total number of fishing vessels"]
        ].sum()

        df_year.rename(columns={
            "Total Fish Landing (Tonnes)": "Landing",
            "Total number of fishing vessels": "Vessels"
        }, inplace=True)

        df_year = df_year[df_year["Landing"] > 0].copy()

        # -------------------------------------------------------------------
        # COORDINATES
        # -------------------------------------------------------------------
        coordinates = {
            "JOHOR TIMUR/EAST JOHORE": [2.0, 104.1],
            "JOHOR BARAT/WEST JOHORE": [1.9, 103.3],
            "JOHOR": [1.4854, 103.7618],
            "MELAKA": [2.1896, 102.2501],
            "NEGERI SEMBILAN": [2.7258, 101.9424],
            "SELANGOR": [3.0738, 101.5183],
            "PAHANG": [3.8126, 103.3256],
            "TERENGGANU": [5.3302, 103.1408],
            "KELANTAN": [6.1254, 102.2381],
            "PERAK": [4.5921, 101.0901],
            "PULAU PINANG": [5.4164, 100.3327],
            "KEDAH": [6.1184, 100.3685],
            "PERLIS": [6.4449, 100.2048],
            "SABAH": [5.9788, 116.0753],
            "SARAWAK": [1.5533, 110.3592],
            "W.P. LABUAN": [5.2831, 115.2308],
        }

        df_year["Coords"] = df_year["State"].map(coordinates)
        df_year = df_year.dropna(subset=["Coords"]).copy()

        # -------------------------------------------------------------------
        # EFFICIENCY
        # -------------------------------------------------------------------
        df_year["Efficiency"] = (
            df_year["Landing"] / df_year["Vessels"].replace(0, np.nan)
        )

        eff_valid = df_year["Efficiency"].dropna()
        eff_avg = eff_valid.mean()

        # Efficiency category thresholds
        q_low, q_mid = eff_valid.quantile([0.33, 0.66])

        def eff_category(x):
            if np.isnan(x):
                return "No Data"
            if x <= q_low:
                return "Low"
            elif x <= q_mid:
                return "Medium"
            else:
                return "High"

        df_year["Eff_Category"] = df_year["Efficiency"].apply(eff_category)

        eff_colors = {
            "High": "#00c853",
            "Medium": "#ffd54f",
            "Low": "#ff5252",
            "No Data": "lightgray",
        }

        # -------------------------
        # SUMMARY CARDS (UPGRADED)
        # -------------------------
 
        with summary_c:
            total_land = df_year["Landing"].sum()
            total_vess = df_year["Vessels"].sum()
            high = df_year.loc[df_year["Landing"].idxmax()]
            low = df_year.loc[df_year["Landing"].idxmin()]

            # BLUE CARD (Normal)
            style_blue = """
                background: linear-gradient(135deg, #00557a 0%, #006b8e 100%);
                padding: 24px;
                border-radius: 18px;
                border: 1px solid rgba(255,255,255,0.14);
                box-shadow: 0 4px 12px rgba(0,0,0,0.35);
            """

            # GREEN CARD (Highest Landing)
            style_green = """
                background: linear-gradient(135deg, #0f7b53 0%, #0a5f46 100%);
                padding: 24px;
                border-radius: 18px;
                border: 1px solid rgba(255,255,255,0.14);
                box-shadow: 0 4px 12px rgba(0,0,0,0.35);
            """

            # RED CARD (Lowest Landing)
            style_red = """
                background: linear-gradient(135deg, #8a1f1f 0%, #a02020 100%);
                padding: 24px;
                border-radius: 18px;
                border: 1px solid rgba(255,255,255,0.14);
                box-shadow: 0 4px 12px rgba(0,0,0,0.35);
            """

            # TEXT STYLE (white contrast)
            text_title = "color:white; font-size:18px; font-weight:500;"
            text_value = "color:white; font-size:34px; font-weight:700;"
            text_state = "color:white; font-size:22px; font-weight:600;"

            col1, col2, col3, col4 = st.columns(4)

            # Total Landing
            with col1:
                st.markdown(f"""
                <div style="{style_blue}">
                    <div style="{text_title}">Total Landing</div>
                    <div style="{text_value}">{total_land:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)

            # Total Vessels
            with col2:
                st.markdown(f"""
                <div style="{style_blue}">
                    <div style="{text_title}">Total Vessels</div>
                    <div style="{text_value}">{total_vess:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)

            # Highest Landing
            with col3:
                st.markdown(f"""
                <div style="{style_green}">
                    <div style="{text_title}">Highest Landing</div>
                    <div style="{text_state}">{high['State']}</div>
                    <div style="{text_value}">{high['Landing']:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)

            # Lowest Landing
            with col4:
                st.markdown(f"""
                <div style="{style_red}">
                    <div style="{text_title}">Lowest Landing</div>
                    <div style="{text_state}">{low['State']}</div>
                    <div style="{text_value}">{low['Landing']:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)


        # -------------------------------------------------------------------
        # STATE FILTER
        # -------------------------------------------------------------------
        with selection_c:
            state_list = sorted(df_year["State"].unique())
            selected_states = st.multiselect("Select State(s):", state_list, default=state_list)

        df = df_year[df_year["State"].isin(selected_states)].copy()

        # -------------------------------------------------------------------
        # MAP THEME
        # -------------------------------------------------------------------
        with selection_c:
            map_theme = st.radio("Choose Map Theme:", ["Light", "Dark", "Satellite", "Default"], horizontal=True)

        tile_map = {
            "Light": "CartoDB positron",
            "Dark": "CartoDB dark_matter",
            "Satellite": "Esri.WorldImagery",
            "Default": "OpenStreetMap",
        }

        # -------------------------------------------------------------------
        # CREATE BASE MAP
        # -------------------------------------------------------------------
        m = folium.Map(location=[4.2, 108.5], zoom_start=6, tiles=None)
        folium.TileLayer(tile_map[map_theme], name="Base Map", control=False).add_to(m)

        lat_min = df["Coords"].apply(lambda x: x[0]).min()
        lat_max = df["Coords"].apply(lambda x: x[0]).max()
        lon_min = df["Coords"].apply(lambda x: x[1]).min()
        lon_max = df["Coords"].apply(lambda x: x[1]).max()

        m.fit_bounds([[lat_min, lon_min], [lat_max, lon_max]])

        # -------------------------------------------------------------------
        # HEATMAP LAYERS
        # -------------------------------------------------------------------
        layer_land = folium.FeatureGroup("Landing Heatmap")
        layer_vess = folium.FeatureGroup("Vessels Heatmap")
        layer_eff = folium.FeatureGroup("Efficiency Heatmap")

        # Landing Heatmap
        HeatMap(
            [[lat, lon, val] for (lat, lon), val in zip(df["Coords"], df["Landing"])],
            radius=40, blur=25, min_opacity=0.4
        ).add_to(layer_land)

        # Vessels Heatmap
        HeatMap(
            [[lat, lon, val] for (lat, lon), val in zip(df["Coords"], df["Vessels"])],
            radius=40, blur=25,
            gradient={0.2:"blue", 0.5:"cyan", 0.7:"lime", 1:"red"}
        ).add_to(layer_vess)

        # -------------------------------------------------------------------
        # EFFICIENCY HEATMAP — FIXED (NO NaN)
        # -------------------------------------------------------------------
        eff_df = df.dropna(subset=["Efficiency"]).copy()

        HeatMap(
            [[lat, lon, val] for (lat, lon), val in zip(eff_df["Coords"], eff_df["Efficiency"])],
            radius=40, blur=30,
            gradient={0.2:"purple", 0.5:"magenta", 0.8:"pink", 1:"white"}
        ).add_to(layer_eff)

        layer_land.add_to(m)
        layer_vess.add_to(m)
        layer_eff.add_to(m)

        # -------------------------------------------------------------------
        # STATE MARKERS (Efficiency Category Colors)
        # -------------------------------------------------------------------
        marker_layer = folium.FeatureGroup("State Markers")

        for _, r in df.iterrows():
            lat, lon = r["Coords"]
            cat = r["Eff_Category"]
            col = eff_colors.get(cat, "lightgray")

            popup = f"""
            <b>{r['State']}</b><br>
            Landing: {r['Landing']:,.0f} t<br>
            Vessels: {r['Vessels']:,.0f}<br>
            Efficiency: {r['Efficiency']:.2f}<br>
            Category: <b>{cat}</b>
            """

            folium.CircleMarker(
                location=[lat, lon],
                radius=10,
                color="black",
                weight=1,
                fill=True,
                fill_color=col,
                fill_opacity=0.9,
                popup=folium.Popup(popup, max_width=250),
                tooltip=r["State"]
            ).add_to(marker_layer)

        marker_layer.add_to(m)

        MiniMap(toggle_display=True).add_to(m)
        Fullscreen().add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)

        # -------------------------------------------------------------------
        # DISPLAY MAP
        # -------------------------------------------------------------------
        with map_c:
            st_folium(m, height=600, width="100%")

        # -------------------------------------------------------------------
        # TABLE
        # -------------------------------------------------------------------
        with table_c:
            st.markdown("### 📊 State Landing / Vessels / Efficiency")
            st.dataframe(
                df.sort_values("Landing", ascending=False).reset_index(drop=True),
                use_container_width=True,
                height=350
            )

        # -------------------------------------------------------------------
        # EFFICIENCY COMPARISON CHART
        # -------------------------------------------------------------------
        with chart_c:
            st.markdown("### 📈 Efficiency Comparison Across States")

            df_chart = df.dropna(subset=["Efficiency"]).sort_values("Efficiency", ascending=False)

            fig, ax = plt.subplots(figsize=(11, 4))
            sns.barplot(
                data=df_chart,
                x="State",
                y="Efficiency",
                hue="Eff_Category",
                palette=eff_colors,
                dodge=False,
                ax=ax
            )

            ax.axhline(eff_avg, linestyle="--", color="gray", linewidth=1.3, label="National Avg")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

            st.pyplot(fig)

        # -------------------------------------------------------------------
        # INTERPRETATION
        # -------------------------------------------------------------------
        with info_c:
            with st.expander("ℹ️ How to interpret the map"):
                st.markdown("""
                ### Layers Explained:
                - **Landing Heatmap (Blue)** – total fish landing  
                - **Vessels Heatmap (Red gradient)** – number of fishing vessels  
                - **Efficiency Heatmap (Purple → White)** – landing per vessel  

                ### Marker Colors:
                - 🟢 High Efficiency  
                - 🟡 Medium  
                - 🔴 Low  

                ### Tips:
                • High Landing + High Efficiency = 🔥 Highly productive region  
                • High Landing + Low Efficiency = ⚠️ Many vessels, low productivity  
                • Low Landing + High Efficiency = 💡 Small fleets but very efficient  
                """)



           
if __name__ == "__main__":
    main()
