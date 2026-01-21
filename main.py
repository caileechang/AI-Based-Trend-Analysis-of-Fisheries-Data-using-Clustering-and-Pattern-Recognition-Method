import os
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
import streamlit.components.v1 as components


# Developer mode flag
DEV_MODE = st.secrets.get("DEV_MODE", "false").lower() == "true"

#DEV_MODE = "STREAMLIT_RUNTIME_VERSION" not in os.environ




# from clustering_method import hierarchical_clustering
@st.cache_data
def load_data():
    print("Loading data...")
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

    # IMPORTANT: allow rows without Month to be counted in YEARLY
    land["Month"] = pd.to_numeric(land["Month"], errors="coerce")

    # YEARLY aggregation: IGNORE Month completely
    yearly_totals = (
        land.groupby(["Year", "State", "Type of Fish"], as_index=False)
            ["Fish Landing (Tonnes)"]
            .sum()
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


    # CLEAN df_vess DIRECTLY 
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


def hdbscan_stability_validation(df, base_mcs, base_ms, X):
    import hdbscan
    import pandas as pd
    import numpy as np

    results = []

    param_grid = [
        (mcs, ms)
        for mcs in range(base_mcs - 1, base_mcs + 2)
        for ms in range(base_ms - 1, base_ms + 2)
        if mcs >= 3 and ms >= 2 and ms <= mcs
    ]

    anomaly_matrix = pd.DataFrame(index=df.index)

    for mcs, ms in param_grid:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,prediction_data=True
        ).fit(X)

        anomaly_matrix[f"{mcs}_{ms}"] = (
            (clusterer.labels_ == -1) |
            (clusterer.outlier_scores_ >
             np.percentile(clusterer.outlier_scores_, 90))
        )

    df["Stability_Score"] = anomaly_matrix.mean(axis=1)

    return df

def dynamic_hdbscan_threshold(df, score_col="Outlier_Norm"):

    # No meaningful separation
    if df[score_col].nunique() <= 1:
        return 1.0

    candidate_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    sens = []
    for t in candidate_thresholds:
        sens.append({
            "threshold": t,
            "count": (df[score_col] >= t).sum()
        })

    sens_df = pd.DataFrame(sens)
    sens_df["delta"] = sens_df["count"].diff().abs()

    # Stable region → conservative choice
    stable = sens_df[sens_df["delta"] == 0]

    if not stable.empty:
        return stable.iloc[-1]["threshold"]

    # Fallback
    return df[score_col].quantile(0.90)

def run_global_hdbscan_outlier_detection(merged_df):
 
    import hdbscan
    from sklearn.preprocessing import StandardScaler

    if merged_df is None or merged_df.empty:
        return pd.DataFrame()

    # Prepare features
    df = merged_df[[
        "State",
        "Year",
        "Total Fish Landing (Tonnes)",
        "Total number of fishing vessels"
    ]].dropna().copy()

    df.rename(columns={
        "Total Fish Landing (Tonnes)": "Landing",
        "Total number of fishing vessels": "Vessels"
    }, inplace=True)

    # Ensure numeric
    df["Landing"] = pd.to_numeric(df["Landing"], errors="coerce")
    df["Vessels"] = pd.to_numeric(df["Vessels"], errors="coerce")
    df = df.dropna(subset=["Landing", "Vessels"])

    if df.shape[0] < 5:
        return df

    # -------------------------
    # Scale features
    # -------------------------
    X = StandardScaler().fit_transform(
        df[["Landing", "Vessels"]]
    )

    # -------------------------
    # AUTO-TUNE HDBSCAN PARAMS
    # -------------------------
    best_params, best_score = auto_tune_hdbscan(
        df,
        min_cluster_range=range(3, 8),
        min_samples_range=range(2, 6)
    )
    # Safe fallback
    if best_params is None:
        min_cluster_size, min_samples = 3, 3
    else:
        min_cluster_size, min_samples = best_params

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        prediction_data=True
    ).fit(X)


# --- GLOSH  ---
    df["Cluster"] = clusterer.labels_         # -1 = noise
    df["Outlier_Score"] = clusterer.outlier_scores_  # GLOSH 

    
    # Normalise 
    
    max_score = df["Outlier_Score"].max()
    df["Outlier_Norm"] = (
        df["Outlier_Score"] / max_score if max_score > 0 else 0.0
    )

    # Dynamic threshold 
   
    chosen_threshold = dynamic_hdbscan_threshold(df)

    df["Anomaly"] = (
        (df["Cluster"] == -1) |                   # density-based anomaly
        (df["Outlier_Norm"] >= chosen_threshold)  # GLOSH-based anomaly
    )

    if DEV_MODE:
        df = hdbscan_stability_validation(
            df,
            min_cluster_size,
            min_samples,
            X
        )

    return df




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

#  Cached wrappers to speed up performance

@st.cache_data
def get_yearly_data(df_land, df_vess):
    return prepare_yearly(df_land, df_vess)

@st.cache_data
def get_monthly_data(df_land, df_vess):
    return prepare_monthly(df_land, df_vess)

@st.cache_data
def get_global_outliers(merged_df):
    return run_global_hdbscan_outlier_detection(merged_df)

def detect_hdbscan_anomalies(df):
    from sklearn.preprocessing import StandardScaler
    import hdbscan
    import numpy as np

    X = df[[
        "Total Fish Landing (Tonnes)",
        "Total number of fishing vessels"
    ]].values

    X_scaled = StandardScaler().fit_transform(X)

    n_samples = X_scaled.shape[0]
    min_cluster_size = max(3, int(np.sqrt(n_samples)))
    min_samples = max(2, int(np.log(n_samples)))

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples
    ).fit(X_scaled)

    labels = clusterer.labels_
    return set(df.index[labels == -1])

def detect_dbscan_anomalies(df):
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors
    from sklearn.cluster import DBSCAN
    from kneed import KneeLocator
    import numpy as np

    X = df[[
        "Total Fish Landing (Tonnes)",
        "Total number of fishing vessels"
    ]].values

    X_scaled = StandardScaler().fit_transform(X)

    n_samples = X_scaled.shape[0]
    min_samples = max(3, int(np.log(n_samples)) + X_scaled.shape[1])

    neigh = NearestNeighbors(n_neighbors=min_samples)
    distances, _ = neigh.fit(X_scaled).kneighbors(X_scaled)
    distances = np.sort(distances[:, min_samples - 1])

    knee = KneeLocator(range(len(distances)), distances,
                       curve="convex", direction="increasing")
    eps = distances[knee.knee] if knee.knee else np.percentile(distances, 90)

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_scaled)

    return set(df.index[labels == -1])

def generate_cluster_label_map(df, cluster_col, landing_col, vessel_col, best_k):
    # Compute cluster scores (higher = better production)
    cluster_stats = (
        df.groupby(cluster_col)[[landing_col, vessel_col]]
        .mean()
        .reset_index()
    )
    cluster_stats["Score"] = cluster_stats[landing_col] + cluster_stats[vessel_col]

    # Sort high → low for score
    cluster_stats = cluster_stats.sort_values("Score", ascending=False)

    # Your label sets
    names_by_k = {
        2: ["High Production", "Low Production"],
        3: ["High Production", "Moderate Production", "Low Production"],
        4: ["High Production", "Marine Driven", "Fleet-driven", "Low"],
        5: ["High", "Fleet-driven", "Moderate", "Low", "Very Low"]
    }

    label_list = names_by_k.get(best_k, names_by_k[3])

    # Create mapping Cluster ID → Label
    label_map = {}
    for i, cluster in enumerate(cluster_stats[cluster_col].values):
        if i < len(label_list):
            label_map[int(cluster)] = label_list[i]
        else:
            label_map[int(cluster)] = f"Cluster {cluster}"
    
    return label_map



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

from scipy.cluster.hierarchy import linkage

def compute_apn_like(Z_full, X_full, best_k):
    """
    APN-like stability metric for hierarchical clustering.
    Lower value = more stable clustering.
    """
    import numpy as np
    from scipy.cluster.hierarchy import fcluster, linkage

    # Full clustering labels
    labels_full = fcluster(Z_full, best_k, criterion="maxclust")

    n_features = X_full.shape[1]
    apn_values = []

    for i in range(n_features):
        # Remove one feature
        X_reduced = np.delete(X_full, i, axis=1)

        # Re-linkage
        Z_reduced = linkage(X_reduced, method="ward")

        # Re-cluster
        labels_reduced = fcluster(Z_reduced, best_k, criterion="maxclust")

        # Proportion of changed assignments
        diff = labels_full != labels_reduced
        apn_values.append(np.mean(diff))

    return np.mean(apn_values)


def hierarchical_clustering(merged_df):

    
    
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
    if DEV_MODE:
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
    # APN-like Stability Metric
    # ----------------------------
    apn_score = compute_apn_like(Z, scaled, best_k)


    # ----------------------------
    # Visualisation Layout
    # ----------------------------
    col1, col2 = st.columns([1, 1])

    # ----------------------------
    # Silhouette Line Chart
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
        if DEV_MODE:
            st.pyplot(fig)

    # ----------------------------
    # Silhouette Table
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

        if DEV_MODE:
            st.dataframe(df_sil, height=230)

    # ----------------------------
    # Best-k Display
    # ----------------------------
    if DEV_MODE:
        st.success(f"Optimal number of clusters (based on Silhouette): **k = {best_k}**")

        with st.expander("📊 Cluster Validation Summary", expanded=False):
            st.markdown(
                f"""
                - **Silhouette Score:** {valid_scores[best_k]:.4f}
                - **APN (stability):** {apn_score:.4f}

                *Lower APN indicates more stable clusters.*
                """
            )

    
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
    st.markdown("---")
    st.dataframe(
        grouped[["State",
                 "Total Fish Landing (Tonnes)",
                 "Total number of fishing vessels",
                 "Cluster"]]
        .sort_values("Cluster")
        .reset_index(drop=True)
    )

def auto_tune_hdbscan(df, min_cluster_range, min_samples_range):
    X = StandardScaler().fit_transform(df[["Landing", "Vessels"]])

    best_score = float("-inf")
    best_params = None

    for mcs in min_cluster_range:
        for ms in min_samples_range:
            if ms > mcs:
                continue

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=mcs,
                min_samples=ms,
                gen_min_span_tree=True
            ).fit(X)

            labels = clusterer.labels_

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters < 2:
                continue

            score = clusterer.relative_validity_

            noise_ratio = np.mean(labels == -1)
            if noise_ratio > 0.5:
                continue

            if score > best_score:
                best_score = score
                best_params = (mcs, ms)

    return best_params, best_score


def run_monthly_hdbscan_outlier_detection(merged_monthly):
 
    import hdbscan
    from sklearn.preprocessing import StandardScaler
    results = []

    if merged_monthly is None or merged_monthly.empty:
        return pd.DataFrame()

    for (year, month), g in merged_monthly.groupby(["Year", "Month"]):

        # HDBSCAN needs enough points
        if g.shape[0] < 5:
            continue

        df = g.copy()

        # Rename for consistency
        df.rename(columns={
            "Fish Landing (Tonnes)": "Landing",
            "Total number of fishing vessels": "Vessels"
        }, inplace=True)

        # Ensure numeric
        df["Landing"] = pd.to_numeric(df["Landing"], errors="coerce")
        df["Vessels"] = pd.to_numeric(df["Vessels"], errors="coerce")
        df = df.dropna(subset=["Landing", "Vessels"])

        if df.shape[0] < 5:
            continue

        # Scale
        X = StandardScaler().fit_transform(df[["Landing", "Vessels"]])

        #  AUTO-TUNE HDBSCAN 
     
        best_params, best_score = auto_tune_hdbscan(
            df,
            min_cluster_range=range(3, 8),
            min_samples_range=range(2, 6)
        )

        # Safe fallback (VERY IMPORTANT)
        if best_params is None:
            min_cluster_size, min_samples = 3, 3
        else:
            min_cluster_size, min_samples = best_params


        #  HDBSCAN (AUTOMATIC)
  
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples
        ).fit(X)

        df["Cluster"] = clusterer.labels_

  
        # MONTHLY ANOMALY RULE (STABLE)
   
        df["Anomaly"] = (df["Cluster"] == -1)

        # Time label
        df["YearMonth"] = f"{int(year)}-{int(month):02d}"

        results.append(df)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)



def main():
    st.set_page_config(layout='wide')

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
    # --- Load base data only once & show a smooth loading UX ---
    if "base_land" not in st.session_state:
        
        st.session_state.base_land, st.session_state.base_vess = load_data()
        st.session_state.data_updated = False


    
  #  df_land = st.session_state.base_land.copy()
    #df_vess = st.session_state.base_vess.copy()
    df_land = st.session_state.base_land
    df_vess = st.session_state.base_vess

   

    def detect_dataset_type(filename):
        name = filename.lower().replace(" ", "").replace("_", "")
        if "landing" in name:
            return "Fish Landing"
        if "vessel" in name or "vessels" in name:
            return "Fish Vessels"
        return None
    
    import re

    def normalize_col(col):
        return re.sub(r'[^a-z0-9]', '', col.lower())
    
    
   
    # Upload additional yearly CSV
    st.sidebar.markdown("### Upload Your Yearly Dataset")
    # uploaded_file = st.sidebar.file_uploader("Upload Excel file only (.xlsx)", type=["xlsx"])
   # uploaded_file = st.sidebar.file_uploader("Upload dataset (.xlsx or .csv)", type=["xlsx", "csv"])

    uploaded_files = st.sidebar.file_uploader(
        "Upload Excel (1 file) or CSV (2 files)",
        type=["xlsx", "csv"],
        accept_multiple_files=True
    )

    user_land = None
    user_vess = None

    if uploaded_files:
        try:
            # ============================
            # CASE 1: Excel (.xlsx)
            # ============================
            if len(uploaded_files) == 1 and uploaded_files[0].name.endswith(".xlsx"):
                uploaded_name = uploaded_files[0].name

                excel = pd.ExcelFile(uploaded_files[0])

                if {"Fish Landing", "Fish Vessels"} <= set(excel.sheet_names):
                    user_land = pd.read_excel(excel, "Fish Landing")
                    user_vess = pd.read_excel(excel, "Fish Vessels")
                    

                else:
                    st.error("Excel must contain sheets: Fish Landing & Fish Vessels")
                    st.stop()

            # ============================
            # CASE 2: CSV (2 files ONLY)
            # ============================
            elif len(uploaded_files) == 2 and all(f.name.endswith(".csv") for f in uploaded_files):

                if "upload_toast_shown" not in st.session_state:
                    st.session_state.upload_toast_shown = False

                user_land, user_vess = None, None
                unresolved = []

                for f in uploaded_files:
                    detected = detect_dataset_type(f.name)

                    if detected == "Fish Landing":
                        user_land = pd.read_csv(f)
                    elif detected == "Fish Vessels":
                        user_vess = pd.read_csv(f)
                    else:
                        unresolved.append(f)

                if unresolved:
                    st.warning("Some files could not be auto-identified. Please assign manually.")

                    assignments = {}
                    for f in unresolved:
                        assignments[f.name] = st.selectbox(
                            f"Dataset type for {f.name}",
                            ["Fish Landing", "Fish Vessels"],
                            key=f.name
                        )

                    for f in unresolved:
                        if assignments[f.name] == "Fish Landing":
                            user_land = pd.read_csv(f)
                        else:
                            user_vess = pd.read_csv(f)

                # 🔒 STRICT: BOTH REQUIRED
                if user_land is None or user_vess is None:
                    st.error(
                        "❌ Analysis requires BOTH datasets:\n"
                        "- Fish Landing CSV\n"
                        "- Fish Vessels CSV"
                    )
                    st.stop()

                if not st.session_state.upload_toast_shown:
                    st.toast(
                        "Both Fish Landing & Fish Vessels CSV files uploaded successfully",
                        icon="✅"
                    )
                    st.session_state.upload_toast_shown = True

            # ============================
            # 🚨 BLOCK ALL OTHER CASES
            # ============================
            else:
                st.error(
                    "❌ Invalid upload.\n\n"
                    "Please upload:\n"
                    "- ONE Excel (.xlsx) with both sheets, OR\n"
                    "- EXACTLY TWO CSV files (Fish Landing + Fish Vessels)."
                )
                st.stop()

        except Exception as e:
                st.error(f"Error reading uploaded file: {e}")

        if user_land is not None and user_vess is not None:
            
                    #st.dataframe(user_land, use_container_width=True, height=400)
                    # VALIDATE UPLOADED DATASET SCHEMA (CRITICAL)
                    # --- Required columns ---
                    required_land_cols = {
                        "state",
                        "year",
                        "month",
                        "typeoffish",
                        "fishlandingtonnes"
                    }

                    required_vess_cols = {
                        "state",
                        "year",
                        "inboardpowered",
                        "outboardpowered",
                        "nonpowered"
                    }

                    pretty_col_names = {
                        "state": "State",
                        "year": "Year",
                        "month": "Month",
                        "typeoffish": "Type of Fish",
                        "fishlandingtonnes": "Fish Landing (Tonnes)",
                        "inboardpowered": "Inboard Powered",
                        "outboardpowered": "Outboard Powered",
                        "nonpowered": "Non-Powered"
                    }

                    # Normalize headers 
                    land_cols = {normalize_col(c) for c in user_land.columns}
                    vess_cols = {normalize_col(c) for c in user_vess.columns}

                    missing_land = required_land_cols - land_cols
                    missing_vess = required_vess_cols - vess_cols

                    if missing_land:
                        missing_land_pretty = [pretty_col_names[c] for c in missing_land]
                        st.error(
                            "❌ Invalid Fish Landing dataset.\n\n"
                            "Missing required column(s):\n"
                            f"{', '.join(missing_land_pretty)}"
                        )
                        st.info(
                            "Fish Landing dataset MUST contain:\n"
                            "- State\n- Year\n- Month\n- Type of Fish\n- Fish Landing (Tonnes)"
                        )
                        st.stop()

                    if missing_vess:
                        missing_vess_pretty = [pretty_col_names[c] for c in missing_vess]

                        st.error(
                            "❌ Invalid Fish Vessels dataset.\n\n"
                            "Missing required column(s):\n"
                            f"{', '.join(missing_vess_pretty)}"
                        )
                        st.info(
                            "Fish Vessels dataset MUST contain:\n"
                            "- State\n- Year\n- Inboard Powered\n- Outboard Powered\n- Non-Powered"
                        )
                        st.stop()

                    # =====================================================
                    # STANDARDISE COLUMN NAMES (AFTER VALIDATION)
                    # =====================================================
                    user_land.columns = [normalize_col(c) for c in user_land.columns]
                    user_vess.columns = [normalize_col(c) for c in user_vess.columns]

                    user_land.rename(columns={
                        "state": "State",
                        "year": "Year",
                        "month": "Month",
                        "typeoffish": "Type of Fish",
                        "fishlandingtonnes": "Fish Landing (Tonnes)"
                    }, inplace=True)

                    user_vess.rename(columns={
                        "state": "State",
                        "year": "Year",
                        "inboardpowered": "Inboard Powered",
                        "outboardpowered": "Outboard Powered",
                        "nonpowered": "Non-Powered"
                    }, inplace=True)


                    msg2=st.info(f"Detected uploaded years: {sorted(user_land['Year'].dropna().unique().astype(int).tolist())}")
                   
        
                    # --- Clean uploaded data to match base format ---
                    #user_land.columns = user_land.columns.str.strip().str.title()
                    user_land['Month'] = user_land['Month'].astype(str).str.strip().str.title()
                    user_land['State'] = user_land['State'].astype(str).str.upper().str.strip()
                    user_land['Type of Fish'] = user_land['Type of Fish'].astype(str).str.title().str.strip()

                    #user_land['Type Of Fish'] = user_land['Type Of Fish'].astype(str).str.title().str.strip()
                    #user_land.rename(columns={'Type Of Fish': 'Type of Fish'}, inplace=True)

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

                    user_land["__source"] = "uploaded"
                    df_land["__source"] = "base"

                    def normalize_month(df):
                        df = df.copy()
                        df["Month"] = pd.to_numeric(df["Month"], errors="coerce")
                        return df

                    df_land = normalize_month(df_land)
                    user_land = normalize_month(user_land)



        # ===== PREVENT RE-MERGING SAME UPLOAD ON RERUN =====
                    upload_key = tuple(sorted([f.name for f in uploaded_files]))

                    if st.session_state.get("last_merged_upload") != upload_key:

                        #df_land = pd.concat([df_land, user_land], ignore_index=True)
                        #  REMOVE BASE DATA THAT CONFLICTS WITH UPLOAD (MONTH-AWARE)
                        df_land["Month_dedup"] = df_land["Month"].fillna(-1)
                        user_land["Month_dedup"] = user_land["Month"].fillna(-1)

                        keys = user_land[['Year', 'State', 'Type of Fish', 'Month_dedup']].drop_duplicates()

                        df_land = df_land.merge(
                            keys,
                            on=['Year', 'State', 'Type of Fish', 'Month_dedup'],
                            how='left',
                            indicator=True
                        )

                        df_land = df_land[df_land['_merge'] == 'left_only'].drop(columns=['_merge', 'Month_dedup'])

                        # Append uploaded data
                        df_land = pd.concat([df_land, user_land.drop(columns='Month_dedup')], ignore_index=True)

                        # NaN-safe deduplication
                        df_land['Month_dedup'] = df_land['Month'].fillna(-1)

                        df_land = (
                            df_land
                            .drop_duplicates(
                                subset=['State', 'Year', 'Month_dedup', 'Type of Fish'],
                                keep='last'
                            )
                            .drop(columns='Month_dedup')
                        )

                        df_vess = (
                            pd.concat([df_vess, user_vess], ignore_index=True)
                            .drop_duplicates(subset=['State', 'Year'], keep='last')
                        )

                        st.session_state.base_land = df_land.copy()
                        st.session_state.base_vess = df_vess.copy()
                        st.session_state.last_merged_upload = upload_key

                    # --- Merge uploaded data with base historical data (SAME structure) ---
                    #df_land = pd.concat([df_land, user_land], ignore_index=True).drop_duplicates(subset=['State', 'Year', 'Month', 'Type of Fish'])
                    
                    

                    #msg1=st.toast(" Uploaded data successfully merged with existing dataset.")
                    
                    # --- Clean uploaded vessel data to match base format ---
                    #user_vess.columns = user_vess.columns.str.strip().str.title()
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

                    #df_vess = pd.concat([df_vess, user_vess], ignore_index=True).drop_duplicates(subset=['State', 'Year'])

                                        # Update session state immediately and keep merged data
                   # st.session_state.base_land = df_land.copy()
                    #st.session_state.base_vess = df_vess.copy()
                    st.session_state.data_updated = True  # mark that new data exists
                    st.cache_data.clear()
                    

                    st.sidebar.success("New dataset merged. Visualizations will refresh automatically.")
                    


        
        

    #merged_df = prepare_yearly(df_land, df_vess)
    merged_df = get_yearly_data(df_land, df_vess)
    




    # =====================================================
    # RUN GLOBAL HDBSCAN ONCE (CACHE IN SESSION STATE)
    # =====================================================
   # if "global_outliers" not in st.session_state or st.session_state.get("data_updated", False):
        #st.session_state.global_outliers = run_global_hdbscan_outlier_detection(merged_df)
       # st.session_state.global_outliers = get_global_outliers(merged_df)
      #  st.session_state.data_updated = False


    #merged_monthly = prepare_monthly(df_land, df_vess)
# Refresh cached monthly summary if new data was uploaded
    if st.session_state.get("data_updated", False):
        st.cache_data.clear()
        st.session_state.data_updated = False

    merged_monthly = get_monthly_data(df_land, df_vess)

    # --- Sidebar for visualization selection ---
    st.sidebar.header("Select Visualization")

    # Base (public) options
    plot_options = [
        "Yearly Fish Landing Summary",
        "Yearly Cluster Trends for Marine and Freshwater Fish",
        "Relationship: Fish Landing vs Fishing Vessels (2D)",
        "Relationship: Fish Landing vs Fishing Vessels (3D)",
        
        "Yearly Fisheries Outlier Detection",
        "Monthly Fisheries Outlier Detection",
        "Fisheries-Based Malaysian States Grouping",
        "Geospatial Maps"
    ]

    # Developer-only option
    if DEV_MODE:
        plot_options.append("Model Stability Test (DBSCAN vs HDBSCAN)")
        plot_options.append("Automatic DBSCAN")

    # Sidebar radio
    plot_option = st.sidebar.radio(
        "Choose a visualization:",
        plot_options
    )

    if plot_option == "Yearly Fish Landing Summary":
           
        import seaborn as sns
        import matplotlib.pyplot as plt
        import numpy as np

        st.markdown("##  Yearly Fish Landing")
         
        

        # --- ALWAYS use cleaned yearly summary from prepare_yearly ---
    #    if uploaded_files:
    #        yearly_summary = prepare_yearly(df_land, df_vess)
      #  else:
         #   yearly_summary = st.session_state.get(
         #       "yearly_summary", prepare_yearly(df_land, df_vess)
          #  )

        #st.session_state.yearly_summary = yearly_summary
        yearly_summary = prepare_yearly(df_land, df_vess)
        st.session_state.yearly_summary = yearly_summary

        # A) Summary Cards + Lollipop 


        # Get latest year
        latest_year = int(yearly_summary["Year"].max())
        available_years = sorted(yearly_summary["Year"].unique())

        selected_year = st.selectbox(
            "Select year to analyse:",
            available_years,
            index=len(available_years) - 1
        )

        filtered_latest = yearly_summary[
            yearly_summary["Year"] == selected_year
        ]

        prev_year = selected_year - 1

        filtered_latest = yearly_summary[yearly_summary["Year"] == selected_year].copy()

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

        
        st.markdown(f"### 🏅 Top 3 States in {selected_year}")

        card_cols = st.columns(3)
        
        
        for idx, (_, row) in enumerate(top3.iterrows()):
            with card_cols[idx]:
                state = row["State"]
                total = row["Total Fish Landing (Tonnes)"]
                prev_val = row["Prev_Year"]
                growth_html = growth_text(total, prev_val)

                card_html = f"""
                <div style="
                    background: linear-gradient(145deg, #0f766e, #022c22);
                    border-radius: 18px;
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


        # CHART FOR LATEST YEAR
   
        st.markdown(f"### Total Fish Landing by State ({selected_year})")

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

      
        #  SHOW YEAR SELECTOR & TABLE
  
        

        filtered_selected = yearly_summary[
            yearly_summary["Year"] == selected_year
        ].sort_values("Total Fish Landing (Tonnes)", ascending=False)

        st.markdown(f"### Fish Landing by State — {selected_year}")
        st.dataframe(filtered_selected, use_container_width=True, height=350)




    elif plot_option == "Yearly Cluster Trends for Marine and Freshwater Fish":

        import matplotlib.pyplot as plt
        import seaborn as sns
        import plotly.graph_objects as go
        import plotly.express as px


        #  CARD STYLE + CHART STYLES
      
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

        st.markdown("## Fish Landing Trends (Cluster Analysis)")

        st.markdown(
            "Compare freshwater & marine fish landings across yearly or monthly periods using K-Means cluster grouping."
        )

        st.markdown("---")


  
        # OPTIONS SECTION
       
        with st.container():

            st.markdown(
                
                "Please select the period and trend to display the fish landing analysis."
                ,
                unsafe_allow_html=True
            )

          
            period_choice = st.radio("Period:", ["Yearly", "Monthly"], horizontal=True)

     
        # HELPER FUNCTIONS FOR CLUSTER MEANINGS
 
 
        def interpret_label(fw, ma, avg_fw, avg_ma):
            """Return High/Low FW & Marine meaning."""
            fw_label = "High Freshwater" if fw >= avg_fw else "Low Freshwater"
            ma_label = "High Marine" if ma >= avg_ma else "Low Marine"
            return f"{fw_label} & {ma_label}"

        def friendly_name(meaning):
            """Convert meaning into user-friendly names."""
            if "High Freshwater" in meaning and "Low Marine" in meaning:
                return "🐟 Freshwater Dominant"
            if "Low Freshwater" in meaning and "High Marine" in meaning:
                return "🌊 Marine Dominant"
            if "High Freshwater" in meaning and "High Marine" in meaning:
                return "🔥 High Production Region"
            if "Low Freshwater" in meaning and "Low Marine" in meaning:
                return "⚠️ Low Production Group"
            return "Mixed Cluster"


        # YEARLY VIEW
       
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

          # -------- YEAR SELECTION DROPDOWN --------
            available_years = sorted(yearly["Year"].unique())
            latest_year = max(available_years)

            selected_year = st.selectbox(
                "Select Year:",
                available_years,
                index=available_years.index(latest_year)
            )

            prev_year = selected_year - 1


            def safe_get(df, year, col):
                row = df.loc[df["Year"] == year, col]
                return row.values[0] if len(row) else 0
            
            
            fw_val = safe_get(yearly, selected_year, "Freshwater (Tonnes)")
            ma_val = safe_get(yearly, selected_year, "Marine (Tonnes)")
            fw_prev = safe_get(yearly, prev_year, "Freshwater (Tonnes)")
            ma_prev = safe_get(yearly, prev_year, "Marine (Tonnes)")

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

            st.markdown(f"## Landing Summary in {selected_year}")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    f"""
                    <div style="{card_style}">
                        <h3 style="color:white;">Freshwater Landing</h3>
                        <h1 style="color:white; font-size:42px;"><b>{fw_val:,.0f}</b> tonnes</h1>
                        {growth_html(fw_val, fw_prev)}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    f"""
                    <div style="{card_style}">
                        <h3 style="color:white;">Marine Landing</h3>
                        <h1 style="color:white; font-size:42px;"><b>{ma_val:,.0f}</b> tonnes</h1>
                        {growth_html(ma_val, ma_prev)}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("---")
    
    
            #  K-MEANS CLUSTERING (REQUIRED FOR PLOTLY)
         
            features = ["Freshwater (Tonnes)", "Marine (Tonnes)"]
            scaled = StandardScaler().fit_transform(yearly[features])

            best_k = st.session_state.get("best_k_yearly", 3)

            yearly["Cluster"] = KMeans(
                n_clusters=best_k,
                random_state=42
            ).fit_predict(scaled)

            st.markdown(f"**Optimal clusters used:** {best_k}")

        
            # PREPARE DATAFRAME FOR PLOTTING
 
            df_plot = yearly.copy()
            df_plot["Freshwater (Tonnes)"] = df_plot["Freshwater (Tonnes)"].astype(float)
            df_plot["Marine (Tonnes)"] = df_plot["Marine (Tonnes)"].astype(float)

            
            # BUILD INTERACTIVE PLOTLY FIGURE (dual axis)
            
            fig = go.Figure()

            # ---- Freshwater Lines ----
            for cl in sorted(df_plot["Cluster"].unique()):
                sub = df_plot[df_plot["Cluster"] == cl]
                fig.add_trace(go.Scatter(
                    x=sub["Year"],
                    y=sub["Freshwater (Tonnes)"],
                    mode="lines+markers",
                    name=f"Freshwater – Cluster {cl}",
                    line=dict(width=2),
                    marker=dict(size=7),
                    hovertemplate="<b>Year:</b> %{x}<br><b>Freshwater:</b> %{y:,.0f} tonnes<extra></extra>",
                ))

            # ---- Marine Lines (right axis) ----
            for cl in sorted(df_plot["Cluster"].unique()):
                sub = df_plot[df_plot["Cluster"] == cl]
                fig.add_trace(go.Scatter(
                    x=sub["Year"],
                    y=sub["Marine (Tonnes)"],
                    mode="lines+markers",
                    name=f"Marine – Cluster {cl}",
                    line=dict(width=2, dash="dash"),
                    marker=dict(size=7),
                    hovertemplate="<b>Year:</b> %{x}<br><b>Marine:</b> %{y:,.0f} tonnes<extra></extra>",
                    yaxis="y2"
                ))

        
            #  Highlight selected year points
           
            fig.add_trace(go.Scatter(
                x=[selected_year],
                y=[fw_val],
                mode="markers",
                marker=dict(size=18, color="white", line=dict(width=2, color="black")),
                name="Selected year – Freshwater"
            ))

            fig.add_trace(go.Scatter(
                x=[selected_year],
                y=[ma_val],
                mode="markers",
                marker=dict(size=18, color="yellow", line=dict(width=2, color="black")),
                name="Selected year – Marine",
                yaxis="y2"
            ))

            # layout settings   
          
            fig.update_layout(
                title=f"Yearly Fish Landing Trends (k={best_k})",
                xaxis=dict(title="Year", tickmode="linear"),
                yaxis=dict(title="Freshwater Landing (Tonnes)", color="blue"),
                yaxis2=dict(
                    title="Marine Landing (Tonnes)",
                    overlaying="y",
                    side="right",
                    color="red"
                ),
                hovermode="x unified",
                template="plotly_white",
                legend=dict(orientation="h", y=-0.25)
            )

           
            #  show chart in streamlit
       
            st.plotly_chart(fig, use_container_width=True)

          
            # Cluster interpretation summary

            st.markdown("## 🔍 Cluster Interpretation Summary")
            

            df_year = df_plot.copy()
            avg_fw = df_year["Freshwater (Tonnes)"].mean()
            avg_ma = df_year["Marine (Tonnes)"].mean()

            cluster_summary = []

            for cl in sorted(df_year["Cluster"].unique()):
                sub = df_year[df_year["Cluster"] == cl]

                fw_mean = sub["Freshwater (Tonnes)"].mean()
                ma_mean = sub["Marine (Tonnes)"].mean()

                if fw_mean >= avg_fw:
                    fw_label = "High Freshwater"
                else:
                    fw_label = "Low Freshwater"

                if ma_mean >= avg_ma:
                    ma_label = "High Marine"
                else:
                    ma_label = "Low Marine"

                if fw_label == "High Freshwater" and ma_label == "High Marine":
                    meaning = "🔥 High Production Region"
                elif fw_label == "High Freshwater" and ma_label == "Low Marine":
                    meaning = "🐟 Freshwater Dominant Region"
                elif fw_label == "Low Freshwater" and ma_label == "High Marine":
                    meaning = "🌊 Marine Dominant Region"
                else:
                    meaning = "⚠️ Low Production Region"

                cluster_summary.append([cl, fw_mean, ma_mean, meaning])

            summary_df = pd.DataFrame(cluster_summary, 
                columns=["Cluster", "Avg Freshwater", "Avg Marine", "Interpretation"])

            st.dataframe(summary_df.style.format({
                "Avg Freshwater": "{:,.2f}",
                "Avg Marine": "{:,.2f}",
            }))


            


        if period_choice == "Monthly":
           
            # Prepare monthly data
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
            # ===============================
            # CLEAN YEAR & MONTH (CRITICAL)
            # ===============================
            monthly["Year"] = pd.to_numeric(monthly["Year"], errors="coerce")
            monthly["Month"] = pd.to_numeric(monthly["Month"], errors="coerce")

            # Keep only valid months
            monthly = monthly[
                (monthly["Month"] >= 1) &
                (monthly["Month"] <= 12)
            ]

            # Drop rows where Year or Month is still NaN
            monthly = monthly.dropna(subset=["Year", "Month"])

            monthly["MonthYear"] = pd.to_datetime(
                monthly["Year"].astype(int).astype(str)
                + "-"
                + monthly["Month"].astype(int).astype(str)
                + "-01",
                errors="coerce"
            )

            # Drop any remaining invalid datetime rows
            monthly = monthly.dropna(subset=["MonthYear"])

            # Create proper datetime column for indexing
            #monthly["MonthYear"] = pd.to_datetime(
            #    monthly["Year"].astype(str) + "-" + monthly["Month"].astype(str) + "-01"
            #)

            # ======================================
            # USER SELECT: YEAR
            # ======================================
            available_years = sorted(monthly["Year"].unique())

            selected_year = st.selectbox(
                "Select Year:",
                available_years,
                index=len(available_years) - 1  # Default to latest year
            )

            # ======================================
            # USER SELECT: MONTH (filtered by year)
            # ======================================
            months_in_year = sorted(
                monthly[monthly["Year"] == selected_year]["Month"].unique()
            )

            month_name_map = {
                1: "January", 2: "February", 3: "March", 4: "April",
                5: "May", 6: "June", 7: "July", 8: "August",
                9: "September", 10: "October", 11: "November", 12: "December"
            }

            month_display = [month_name_map[m] for m in months_in_year]

            selected_month_name = st.selectbox(
                "Select Month:",
                month_display
            )

            selected_month = {v: k for k, v in month_name_map.items()}[selected_month_name]

            # Selected month-year
            selected_date = pd.to_datetime(f"{selected_year}-{selected_month}-01")
            prev_date = selected_date - pd.DateOffset(months=1)

            # Helper to get values safely
            def safe_month_value(df, date, col):
                v = df.loc[df["MonthYear"] == date, col]
                return v.values[0] if len(v) else 0

            # ======================================
            # IMPROVED GROWTH FORMULA (SAME AS YEARLY)
            # ======================================
            def calc_growth_month_html(curr, prev):
                try:
                    prev = float(prev)
                    curr = float(curr)
                except:
                    return "<span style='color:gray;'>–</span>"

                # No previous OR no current → show dash
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

            # ======================================
            # GET CURRENT & PREVIOUS VALUES
            # ======================================
            fw = safe_month_value(monthly, selected_date, "Freshwater (Tonnes)")
            fw_prev = safe_month_value(monthly, prev_date, "Freshwater (Tonnes)")
            ma = safe_month_value(monthly, selected_date, "Marine (Tonnes)")
            ma_prev = safe_month_value(monthly, prev_date, "Marine (Tonnes)")

            # ======================================
            # PREMIUM SUMMARY CARDS
            # ======================================

            card_style = """
                background: linear-gradient(135deg, #06373d 0%, #001f24 100%);
                padding: 30px 35px;
                border-radius: 20px;
                border: 1.2px solid rgba(0, 255, 200, 0.25);
                box-shadow: 0 0 18px rgba(0, 255, 200, 0.12);
            """

            st.markdown(f"## Landing Summary in {selected_month_name} {selected_year}")

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


            # ============================================================
            # K-MEANS CLUSTERING
            # ============================================================
            features = ["Freshwater (Tonnes)", "Marine (Tonnes)"]
            scaled = StandardScaler().fit_transform(monthly[features])
            best_k = st.session_state.get("best_k_monthly", 3)

            monthly["Cluster"] = KMeans(n_clusters=best_k, random_state=42).fit_predict(scaled)

            st.markdown(f"**Optimal clusters used:** {best_k}")

            # Melt for plotting (long format)
            melted = monthly.melt(
               id_vars=["MonthYear", "Cluster"],
               value_vars=["Freshwater (Tonnes)", "Marine (Tonnes)"],
               var_name="Type",
               value_name="Landing",
         )

            # Shared linestyles
            linestyles = ["solid", "dashed", "dotted", "dashdot"]

            # ============================================================
            # MONTHLY TREND (PLOTLY REPLACEMENT FOR MATPLOTLIB)
            # ============================================================

            import plotly.graph_objects as go
            import plotly.express as px
            import numpy as np
            


            # Prepare monthly dataframe for selected year
            dfm = monthly.copy()
            dfm["MonthYear"] = pd.to_datetime(dfm["MonthYear"])
            dfm["MonthIndex"] = dfm["MonthYear"].dt.month

            dfm_selected = dfm[dfm["Year"] == selected_year].copy()

            # =============== KMEANS ===============
            features = ["Freshwater (Tonnes)", "Marine (Tonnes)"]
            scaled = StandardScaler().fit_transform(dfm_selected[features])
            best_k = st.session_state.get("best_k_monthly", 3)

            dfm_selected["Cluster"] = KMeans(n_clusters=best_k, random_state=42).fit_predict(scaled)
            

            # =============== BUILD PLOTLY FIGURE ===============
            fig = go.Figure()

            colors_fw = ["#1f77b4", "#2ca02c", "#17becf", "#9467bd"]
            colors_ma = ["#d62728", "#ff7f0e", "#e377c2", "#8c564b"]

            # ------------------------------------
            # FRESHWATER LINES
            # ------------------------------------
            for cl in sorted(dfm_selected["Cluster"].unique()):
                sub = dfm_selected[dfm_selected["Cluster"] == cl]

                fig.add_trace(go.Scatter(
                    x=sub["MonthIndex"],
                    y=sub["Freshwater (Tonnes)"],
                    mode="lines+markers",
                    name=f"Freshwater – Cluster {cl}",
                    line=dict(width=3, color=colors_fw[cl % len(colors_fw)]),
                    marker=dict(size=9),
                    hovertemplate="<b>Month:</b> %{x}<br><b>Freshwater:</b> %{y:,.0f} tonnes<extra></extra>"
                ))

            # ------------------------------------
            # MARINE LINES
            # ------------------------------------
            for cl in sorted(dfm_selected["Cluster"].unique()):
                sub = dfm_selected[dfm_selected["Cluster"] == cl]

                fig.add_trace(go.Scatter(
                    x=sub["MonthIndex"],
                    y=sub["Marine (Tonnes)"],
                    mode="lines+markers",
                    name=f"Marine – Cluster {cl}",
                    line=dict(width=3, color=colors_ma[cl % len(colors_ma)]),
                    marker=dict(size=9),
                    hovertemplate="<b>Month:</b> %{x}<br><b>Marine:</b> %{y:,.0f} tonnes<extra></extra>",
                    yaxis="y2"
                ))

            # =============== HIGHLIGHT SELECTED MONTH ===============
            fw_val = dfm_selected.loc[dfm_selected["MonthYear"] == selected_date, "Freshwater (Tonnes)"]
            ma_val = dfm_selected.loc[dfm_selected["MonthYear"] == selected_date, "Marine (Tonnes)"]

            if len(fw_val):
                fig.add_trace(go.Scatter(
                    x=[selected_month],
                    y=[fw_val.values[0]],
                    mode="markers",
                    marker=dict(size=18, color="white", line=dict(width=2, color="black")),
                    name="Selected Month – Freshwater"
                ))

            if len(ma_val):
                fig.add_trace(go.Scatter(
                    x=[selected_month],
                    y=[ma_val.values[0]],
                    mode="markers",
                    marker=dict(size=18, color="yellow", line=dict(width=2, color="black")),
                    name="Selected Month – Marine",
                    yaxis="y2"
                ))

            # =============== LAYOUT ===============
            fig.update_layout(
                title=f"Monthly Fish Landing Trends ({selected_year})",
                xaxis=dict(
                    title="Month",
                    tickmode="array",
                    tickvals=list(range(1, 13)),
                    ticktext=["Jan","Feb","Mar","Apr","May","Jun","Jul",
                            "Aug","Sep","Oct","Nov","Dec"]
                ),
                yaxis=dict(title="Freshwater (Tonnes)", color="blue"),
                yaxis2=dict(
                    title="Marine (Tonnes)",
                    overlaying="y",
                    side="right",
                    color="red"
                ),
                hovermode="x unified",
                template="plotly_white",
                legend=dict(orientation="h", y=-0.2)
            )

            st.plotly_chart(fig, use_container_width=True)


            # CLUSTER INTERPRETATION 
        
            st.markdown("## 🔍 Cluster Interpretation Summary")

            # Compute averages
            avg_fw = dfm_selected["Freshwater (Tonnes)"].mean()
            avg_ma = dfm_selected["Marine (Tonnes)"].mean()

            # Prepare interpretation table
            cluster_summary = []

            for cl in sorted(dfm_selected["Cluster"].unique()):
                sub = dfm_selected[dfm_selected["Cluster"] == cl]

                fw_mean = sub["Freshwater (Tonnes)"].mean()
                ma_mean = sub["Marine (Tonnes)"].mean()

                # Determine freshwater meaning
                if fw_mean >= avg_fw:
                    fw_label = "High Freshwater"
                else:
                    fw_label = "Low Freshwater"

                # Determine marine meaning
                if ma_mean >= avg_ma:
                    ma_label = "High Marine"
                else:
                    ma_label = "Low Marine"

                # Friendly interpretation name
                if fw_label == "High Freshwater" and ma_label == "High Marine":
                    meaning = "🔥 High Production Region"
                elif fw_label == "High Freshwater" and ma_label == "Low Marine":
                    meaning = "🐟 Freshwater Dominant Region"
                elif fw_label == "Low Freshwater" and ma_label == "High Marine":
                    meaning = "🌊 Marine Dominant Region"
                else:
                    meaning = "⚠️ Low Production Region"

                cluster_summary.append([cl, fw_mean, ma_mean, meaning])

            # Display table
            summary_df = pd.DataFrame(cluster_summary, 
                columns=["Cluster", "Avg Freshwater", "Avg Marine", "Interpretation"])

            st.dataframe(summary_df.style.format({
                "Avg Freshwater": "{:,.2f}",
                "Avg Marine": "{:,.2f}",
            }))




    elif plot_option == "Optimal K for Monthly & Yearly":

        

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
        # MONTHLY COMPOSITION
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
        # YEARLY COMPOSITION
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
        #   SUMMARY TABLE
        # ============================================================
        st.markdown("### 🧾 Summary of Optimal K Results") 

        summary = []

        if best_k_monthly is not None:
            summary.append([
                "Monthly",
                best_k_monthly,
                f"{best_sil_monthly:.3f}" if best_sil_monthly else "N/A"
            ])

        if best_k_yearly is not None:
            summary.append([
                "Yearly ",
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


        # ------------------------------------------------------------
        # Enhanced Explanation Box
        # ------------------------------------------------------------
        with st.expander("ℹ️ Why This Page Matters (Click to Expand)"):
            st.markdown("""
            The number of clusters (K) affects how states or months are grouped into meaningful 
            categories such as **High**, **Medium**, and **Low** fish landing performance.

            A correctly chosen **K ensures** that all cluster-based visualisations remain:
            - accurate  
            - consistent  
            - scientifically valid  

            ### 🔍 How This Helps Your Analysis
            - ✔ Ensures your clusters are not too many (noise) or too few (overgeneralised)
            - ✔ Helps you understand which years/states are strong in **freshwater vs marine**
            - ✔ Makes downstream visualisations easier to interpret
        
            """)

        with st.expander(" Understanding Silhouette Score vs Elbow Method"):
            st.markdown("""
            **Silhouette Score:**  
            - Measures how well-separated clusters are  
            - Higher score = better cluster quality  

            **Elbow Method (Inertia):**  
            - Measures how compact clusters are  
            - The "elbow" shows diminishing improvement  

            Using both methods together helps select the most **statistically valid K**.
            """)

    elif plot_option =="Relationship: Fish Landing vs Fishing Vessels (2D)":
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import silhouette_score
        st.subheader("Automatic 2D K-Means Clustering")
    
        # ---  Prepare data ---
        features = merged_df[['Total Fish Landing (Tonnes)', 'Total number of fishing vessels']]
        scaled = StandardScaler().fit_transform(features)
        
        # ---  Compute inertia (Elbow) and silhouette for k = 2–10 ---
        ks = range(2, 11)
        inertia = []
        silhouette = []
    
        for k in ks:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(scaled)
            inertia.append(kmeans.inertia_)
            sil = silhouette_score(scaled, labels)
            silhouette.append(sil)
    
        # ---  Determine the best k (highest silhouette) ---
        best_k = ks[np.argmax(silhouette)]

        if DEV_MODE:
    
            # --- Plot both metrics side by side ---
            
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
    
    
        # Step 6: Human-readable cluster labels
      
        cluster_label_map = generate_cluster_label_map(
            merged_df,
            "Cluster",
            "Total Fish Landing (Tonnes)",
            "Total number of fishing vessels",
            best_k
        )

        merged_df["Cluster_Label"] = merged_df["Cluster"].map(cluster_label_map)


        if DEV_MODE:
        # --- Step 6: Display summary ---
            st.success(f"Optimal number of clusters automatically determined: **k = {best_k}**")
        
            st.markdown("Clusters below are determined automatically based on the **highest Silhouette Score** and Elbow consistency.")


        # ==================================================
        # CORRELATION INSIGHT (Pearson r)
        # ==================================================

        x = merged_df["Total number of fishing vessels"]
        y = merged_df["Total Fish Landing (Tonnes)"]

        pearson_r = x.corr(y)

        if pd.isna(pearson_r):
            direction = "Insufficient data"
            emoji = "⚠️"
            color = "#f59e0b"
            strength = "N/A"
            explanation = (
                "Correlation could not be computed reliably due to limited "
                "data variation."
            )
        else:
            abs_r = abs(pearson_r)

            if abs_r >= 0.8:
                strength = "Very strong"
            elif abs_r >= 0.6:
                strength = "Strong"
            elif abs_r >= 0.4:
                strength = "Moderate"
            elif abs_r >= 0.2:
                strength = "Weak"
            else:
                strength = "Very weak"

            if pearson_r > 0:
                direction = "Positive ↑"
                emoji = "📈"
                color = "#2ecc71"
                explanation = (
                    "A positive linear association is observed between fish vessel "
                    "and fish landings. Both variables tend to increase together."
                )
            elif pearson_r < 0:
                direction = "Negative ↓"
                emoji = "📉"
                color = "#e74c3c"
                explanation = (
                    "An inverse linear association is observed between fish vessel  "
                    "and fish landings. As one increases, the other tends to decrease."
                )
            else:
                direction = "No relationship"
                emoji = "➖"
                color = "#9ca3af"
                explanation = (
                    "No meaningful linear relationship is observed between the variables."
                )

        components.html(
            f"""
            <div style="
                 background: #2563EB;
                border-radius: 24px;
               
                padding: 24px;
                margin-bottom: 22px;

                

               

                
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;

                max-width: 900px;
            ">

                <h3 style="margin:0; font-size: 22px;color:white; font-weight: 700;display: flex; align-items: center;gap: 10px">
                    {emoji} Relationship Insight
                </h3>

                <p style="margin-top:16px; font-size:18px; color:white;">
                    <b>Correlation:</b>
                    <span style="color:{color}; font-weight:700;">
                        {direction}
                    </span>
                </p>

                <p style="font-size:18px; color:white;">
                    <b>Pearson correlation coefficient:</b> {pearson_r:.2f} ({strength})
                </p>

                <p style="font-size:16px; color:white;">
                    {explanation}
                </p>

            </div>
            """
           
        )
    
        # --- Step 7: Show 2D scatter ---
        fig2, ax = plt.subplots(figsize=(10, 6))
        
        sns.scatterplot(
            data=merged_df,
            x='Total number of fishing vessels',
            y='Total Fish Landing (Tonnes)',
            hue='Cluster_Label',
            palette='Set2',
            s=70,
            ax=ax
        )
        ax.set_title("Automatic 2D K-Means Clustering")
        ax.set_xlabel("Total Number of Fishing Vessels")
        ax.set_ylabel("Total Fish Landing (Tonnes)")
        ax.grid(alpha=0.25)

        st.pyplot(fig2)

        # ==================================================
        # Cluster Summary (Improved Column Naming)
        # ==================================================
        summary = (
            merged_df
            .groupby('Cluster_Label')[[ 
                'Total Fish Landing (Tonnes)',
                'Total number of fishing vessels'
            ]]
            .mean()
            .round(0)
            .reset_index()
        )

        # Rename to reflect what the values represent
        summary = summary.rename(columns={
            'Total Fish Landing (Tonnes)': 'Avg Fish Landing (Tonnes)',
            'Total number of fishing vessels': 'Avg Fishing Vessels'
        })

        st.markdown("### 📊 Cluster Summary")
        st.dataframe(summary, use_container_width=True)


    elif plot_option == "Relationship: Fish Landing vs Fishing Vessels (3D)":
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        import plotly.express as px
        from sklearn.metrics import silhouette_score
        
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
        

       
        #st.markdown(f"<small><b>Optimal number of clusters: {best_k}</b></small>", unsafe_allow_html=True)

        #st.markdown("Clusters selected automatically using the highest Silhouette score.")
        cluster_label_map = generate_cluster_label_map(
            merged_df,
            "Cluster",
            "Total Fish Landing (Tonnes)",
            "Total number of fishing vessels",
            best_k
        )
        merged_df["Cluster_Label"] = merged_df["Cluster"].map(cluster_label_map)


        # Color palette matched to number of clusters
        base_colors = ["#FF6D00", "#00E676", "#4FC3F7", "#9575CD", "#E91E63"]
        color_map = {cluster_label_map[c]: base_colors[i] for i, c in enumerate(cluster_label_map.keys())}

        # ===================================================
        # STATIC VERSION 
        # ===================================================
       
        if vis_mode == "Static":
            st.sidebar.markdown("### Adjust 3D View")
            elev = st.sidebar.slider("Vertical tilt", 0, 90, 30)
            azim = st.sidebar.slider("Horizontal rotation", 0, 360, 45)

            plt.close('all')

            # Prepare dataframe for plotting
            df = merged_df.copy()
            df["Landing"] = pd.to_numeric(df["Total Fish Landing (Tonnes)"], errors="coerce")
            df["Vessels"] = pd.to_numeric(df["Total number of fishing vessels"], errors="coerce")
            df = df.dropna(subset=["Landing", "Vessels"])

            # Use same cluster label logic as Interactive version
            df["Cluster_Label"] = df["Cluster"].map(cluster_label_map)

            # Colors match interactive view
            cluster_colors = {label: color_map[label] for label in df["Cluster_Label"].unique()}

            fig = plt.figure(figsize=(7, 6), dpi=120)
            ax = fig.add_subplot(111, projection='3d')

            # Plot each cluster separately for legend support
            for label in df["Cluster_Label"].unique():
                sub = df[df["Cluster_Label"] == label]
                ax.scatter(
                    sub['Vessels'],
                    sub['Landing'],
                    sub['Year'],
                    c=cluster_colors[label],
                    s=40,
                    alpha=0.85,
                    edgecolor="white",
                    linewidth=0.5,
                    label=label  # Adds legend entry
                )

            # Modern styling
            ax.set_facecolor("#F5F5F5")
            ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.3)

            ax.set_xlabel("Vessels", fontsize=8, labelpad=8)
            ax.set_ylabel("Landings", fontsize=8, labelpad=8)
            ax.set_zlabel("Year", fontsize=8, labelpad=8)

            ax.view_init(elev=elev, azim=azim)

            # Title
            ax.set_title(
                f"Static 3D KMeans (k={best_k})",
                fontsize=10,
                weight="bold",
                pad=12,
                color="#222"
            )

            # Add a clean legend
            ax.legend(
                title="Cluster",
                fontsize=7,
                title_fontsize=8,
                frameon=True,
                facecolor="white"
            )

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)


            

        # ===================================================
        # INTERACTIVE VERSION — PLOTLY (FULL 3D ROTATION)
        # ===================================================
       
        else:

            # Prepare dataframe for clustering
            df = merged_df.copy()
            df["Landing"] = pd.to_numeric(df["Total Fish Landing (Tonnes)"], errors="coerce")
            df["Vessels"] = pd.to_numeric(df["Total number of fishing vessels"], errors="coerce")
            df = df.dropna(subset=["Landing", "Vessels"])

            # ================================================
            # DYNAMIC CLUSTER INTERPRETATION BASED ON k
            # ================================================

            # Unified & consistent cluster naming
            cluster_label_map = generate_cluster_label_map(
                df,
                "Cluster",
                "Landing",
                "Vessels",
                best_k
            )
            df["Cluster_Label"] = df["Cluster"].map(cluster_label_map)

            # Color mapping
            base_colors = ["#FF6D00", "#00E676", "#4FC3F7", "#9575CD", "#E91E63"]
            color_map = {
                label: base_colors[i]
                for i, label in enumerate(cluster_label_map.values())
            }

            

            # ================================================
            # PLOT INTERACTIVE 3D CLUSTER RESULTS
            # ================================================
            fig = px.scatter_3d(
                df,
                x='Vessels',
                y='Landing',
                z='Year',
                color='Cluster_Label',
                symbol='Cluster_Label',
                hover_name='State',
                hover_data={'Landing': ':,.0f', 'Vessels': ':,.0f', 'Year': True},
                color_discrete_map=color_map,
                title=f"Interactive 3D KMeans Clustering (k={best_k})",
                height=600,
                opacity=0.85
            )

            fig.update_traces(
                marker=dict(
                    size=6,
                    line=dict(width=0.8, color='black')
                )
            )

            fig.update_layout(
                legend_title="Cluster Category",
                paper_bgcolor='#111111',
                font_color='white',
                coloraxis_showscale=False,  # remove bad color bar
                scene=dict(
                    xaxis_title="Fishing Vessels",
                    yaxis_title="Fish Landing (Tonnes)",
                    zaxis_title="Year",
                    xaxis=dict(backgroundcolor='#1f1f1f'),
                    yaxis=dict(backgroundcolor='#1f1f1f'),
                    zaxis=dict(backgroundcolor='#1f1f1f'),
                ),
                margin=dict(l=0, r=0, b=0, t=50)
            )

            st.plotly_chart(fig, use_container_width=True)

            # ================================================
            # CLUSTER SUMMARY TABLE
            # ================================================
            st.markdown("### Cluster Interpretation Summary")

            summary = (
                df.groupby("Cluster_Label")[["Landing", "Vessels"]]
                .mean()
                .rename(columns={"Landing": "Avg Landing", "Vessels": "Avg Vessels"})
                .reset_index()
            )

            st.dataframe(summary.style.format({
                "Avg Landing": "{:,.0f}",
                "Avg Vessels": "{:,.0f}"
            }), use_container_width=True)

        

    elif plot_option == "Yearly Fisheries Outlier Detection":
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import silhouette_score
        from scipy.spatial import ConvexHull

        #  Run anomaly detection only when needed
        if "global_outliers" not in st.session_state or st.session_state.get("data_updated", False):
            with st.spinner("🔍 Detecting anomalies... Please wait."):
                st.session_state.global_outliers = get_global_outliers(merged_df)
                st.session_state.data_updated = False

        df_global_outliers = st.session_state.global_outliers

        st.subheader("Automatic HDBSCAN Clustering & Outlier Detection")
        #st.dataframe(df_global_outliers)
        # -----------------------------
        # 1. FILTER VALID STATES
        # -----------------------------
        valid_states = [
            "JOHOR TIMUR/EAST JOHORE", "JOHOR BARAT/WEST JOHORE", "JOHOR",
            "MELAKA", "NEGERI SEMBILAN", "SELANGOR", "PAHANG", "TERENGGANU",
            "KELANTAN", "PERAK", "PULAU PINANG", "KEDAH", "PERLIS",
            "SABAH", "SARAWAK", "W.P. LABUAN"
        ]

        df = merged_df[merged_df["State"].isin(valid_states)].copy()

        if df.empty:
            st.warning("No valid data after filtering states.")
            st.stop()

        #  PREPARE FEATURES
        X = df[[
            "Total Fish Landing (Tonnes)",
            "Total number of fishing vessels"
        ]].values

        X_scaled = StandardScaler().fit_transform(X)

        n_samples = X_scaled.shape[0]

        # AUTO HDBSCAN PARAMETERS
   
        min_cluster_size = max(3, int(np.sqrt(n_samples)))
        min_samples = max(2, int(np.log(n_samples)))

        if DEV_MODE:  
            st.markdown(f"**Auto min_cluster_size:** `{min_cluster_size}`")
            st.markdown(f"**Auto min_samples:** `{min_samples}`")


        # RUN HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            prediction_data=True
        )

        labels = clusterer.fit_predict(X_scaled)
        df["HDBSCAN_Label"] = labels
        df["Outlier_Score"] = clusterer.outlier_scores_

    
        mask = labels != -1
        if DEV_MODE:
            if len(set(labels[mask])) > 1:
                    sil = silhouette_score(X_scaled[mask], labels[mask])
                    st.info(f"Silhouette Score (clusters only): `{sil:.3f}`")
            else:
                    st.warning("Silhouette unavailable — only one cluster detected.")

        # CLUSTER VISUALISATION
   
        fig, ax = plt.subplots(figsize=(10, 6))
        unique_labels = sorted(set(labels))
        palette = sns.color_palette("tab10", len(unique_labels))

        for label in unique_labels:
            pts = X_scaled[labels == label]

            if label == -1:
                ax.scatter(
                    pts[:, 1], pts[:, 0],
                    s=45, c="lightgray", edgecolor="k",
                    alpha=0.6, label="Outliers"
                )
            else:
                color = palette[label % len(palette)]
                ax.scatter(
                    pts[:, 1], pts[:, 0],
                    s=65, c=[color], edgecolor="k",
                    alpha=0.85, label=f"Cluster {label} ({len(pts)})"
                )

                # Convex hull
                if len(pts) >= 3:
                    hull = ConvexHull(pts)
                    hv = list(hull.vertices) + [hull.vertices[0]]
                    ax.plot(
                        pts[hv, 1], pts[hv, 0],
                        color=color, linewidth=2
                    )

        ax.set_title("HDBSCAN Clusters & Outliers (Automatic)")
        ax.set_xlabel("Vessels (scaled)")
        ax.set_ylabel("Landings (scaled)")
        ax.grid(alpha=0.3)
        ax.legend()
        st.pyplot(fig)

       
        # CLUSTER SUMMARY
       
        cluster_summary = (
            df[df["HDBSCAN_Label"] != -1]
            .groupby("HDBSCAN_Label")[[
                "Total Fish Landing (Tonnes)",
                "Total number of fishing vessels"
            ]]
            .mean()
            .reset_index()
        )

        # ---- Baseline from cluster centroids  ----

        cluster_summary["Landing per Vessel"] = (
            cluster_summary["Total Fish Landing (Tonnes)"] /
            cluster_summary["Total number of fishing vessels"]
        )

        median_lpv = cluster_summary["Landing per Vessel"].median()
        avg_land   = cluster_summary["Total Fish Landing (Tonnes)"].mean()

        def interpret_cluster(row):
            land = row["Total Fish Landing (Tonnes)"]
            lpv  = row["Landing per Vessel"]

            if land >= avg_land and lpv >= median_lpv:
                return "🚢 High Fish Production & Efficient Fish Vessels"

            if land < avg_land and lpv >= median_lpv:
                return "⚡ Efficient Small-Scale Vessels"

            if land >= avg_land and lpv < median_lpv:
                return "⚠️ Overcapacity Region"

            return "🛶 Low Activity Region"

        
        


        cluster_summary["Cluster Meaning"] = cluster_summary.apply(
            interpret_cluster, axis=1
        )

        # ---- Human-readable cluster name ----
        cluster_summary["Cluster Name"] = (
            "Cluster " +
            cluster_summary["HDBSCAN_Label"].astype(str) +
            " – " +
            cluster_summary["Cluster Meaning"]
        )

        # ---- Display table (hide raw label for users) ----
        st.markdown("### 📊 Cluster Summary")

        display_cluster_summary = cluster_summary[[
            "Cluster Name",
            "Total Fish Landing (Tonnes)",
            "Total number of fishing vessels"
        ]]

        st.dataframe(
            display_cluster_summary.style.format({
                "Total Fish Landing (Tonnes)": "{:,.2f}",
                "Total number of fishing vessels": "{:,.0f}"
            }),
            use_container_width=True
        )

        # =====================================================
        # GLOBAL HDBSCAN + STABILITY (RUN ONCE)
        # =====================================================
        if "global_outliers" not in st.session_state or st.session_state.data_updated:

            base_df = run_global_hdbscan_outlier_detection(merged_df)

            X = base_df[[
                "Total Fish Landing (Tonnes)",
                "Total number of fishing vessels"
            ]].values

            base_df = hdbscan_stability_validation(
                base_df,
                base_mcs=8,
                base_ms=5,
                X=X
            )

            st.session_state.global_outliers = base_df
            st.session_state.data_updated = False

        
        # OUTLIER ANALYSIS
      
        df = st.session_state.global_outliers.copy()

        outliers = df[df["Anomaly"] == True]
        st.success(f"Detected {len(outliers)} outliers.")


        if not outliers.empty:

            avg_land = df["Landing"].mean()
            avg_ves  = df["Vessels"].mean()


            def explain(r):
                if r["Landing"] > avg_land and r["Vessels"] < avg_ves:
                    return "⚠️ High landing but low vessels"
                if r["Landing"] < avg_land and r["Vessels"] > avg_ves:
                    return "🐟 Low catch per vessel"
                if r["Landing"] < avg_land and r["Vessels"] < avg_ves:
                    return "🛶 Low activity region"
                return "Atypical pattern"


            outliers["Why Flagged"] = outliers.apply(explain, axis=1)


            st.markdown("### 🚨 Outlier Details")
           

            if DEV_MODE:
                st.dataframe(outliers, use_container_width=True)
            else:
                st.dataframe(
                    outliers.drop(
                        columns=["HDBSCAN_Label", "Outlier_Score","Outlier_Norm", "Stability_Score"],
                        errors="ignore"
                    ),
                    use_container_width=True
                )



            

            if DEV_MODE:
                st.subheader(" HDBSCAN Outlier Stability Validation")

                # Classify stability strength
                def stability_label(v):
                    if v >= 0.8:
                        return "High (Robust)"
                    elif v >= 0.4:
                        return "Medium (Borderline)"
                    else:
                        return "Low (Unstable)"

                df["Stability_Level"] = df["Stability_Score"].apply(stability_label)

                st.dataframe(
                    df[
                        ["State", "Year", "Landing", "Vessels",
                        "Anomaly", "Stability_Score", "Stability_Level"]
                    ]
                    .sort_values("Stability_Score", ascending=False),
                    use_container_width=True
                )

                st.info(
                    "Stability Score = proportion of parameter-perturbed HDBSCAN runs "
                    "where the point remains anomalous."
                )

   





    
        

 


   


    elif plot_option == "Automatic DBSCAN":
        import numpy as np  
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy.spatial import ConvexHull
        from sklearn.metrics import silhouette_score

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
                #  JITTER ONLY FOR NOISE (visual clarity)
                jitter = np.random.normal(0, 0.02, pts.shape)
                pts_jittered = pts + jitter

                ax.scatter(
                    pts_jittered[:, 1],   # vessels (scaled + jitter)
                    pts_jittered[:, 0],   # landings (scaled + jitter)
                    s=50,
                    c="lightgray",
                    edgecolor="k",
                    alpha=0.6,
                    label="Noise"
                )

            else:
                color = palette[label % len(palette)]
                ax.scatter(
                    pts[:, 1],   # vessels (scaled)
                    pts[:, 0],   # landings (scaled)
                    s=60,
                    c=[color],
                    edgecolor="k",
                    alpha=0.85,
                    label=f"Cluster {label} ({len(pts)})"
                )

                # Convex Hull (NO jitter here – keep geometry exact)
                if len(pts) >= 3:
                    hull = ConvexHull(pts)
                    hv = list(hull.vertices) + [hull.vertices[0]]
                    ax.plot(
                        pts[hv, 1],
                        pts[hv, 0],
                        color=color,
                        linewidth=2
                    )

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

            outliers_debug = outliers.copy()
            outliers_debug["Vessels_scaled"] = scaled[labels == -1][:, 1]
            outliers_debug["Landing_scaled"] = scaled[labels == -1][:, 0]

            st.dataframe(outliers_debug[
                ["Year", "State", "Vessels_scaled", "Landing_scaled"]
            ])

   
    elif plot_option == "Model Stability Test (DBSCAN vs HDBSCAN)":
        if not DEV_MODE:
            st.error("Access denied.")
            st.stop()
        import numpy as np
        import matplotlib.pyplot as plt
        from itertools import combinations

        st.subheader(" Stability Test: DBSCAN vs HDBSCAN")

        def jaccard(A, B):
            if len(A | B) == 0:
                return 1.0
            return len(A & B) / len(A | B)

        def stability_test(df, detector, runs=10, drop_frac=0.1):
            rng = np.random.default_rng(42)
            anomaly_sets = []

            for _ in range(runs):
                drop_n = int(len(df) * drop_frac)
                drop_idx = rng.choice(df.index, drop_n, replace=False)
                df_sub = df.drop(drop_idx)

                anomaly_sets.append(detector(df_sub))

            scores = [
                jaccard(A, B)
                for A, B in combinations(anomaly_sets, 2)
            ]
            return np.mean(scores), scores
        df_eval = merged_df.copy()

        dbscan_mean, dbscan_scores = stability_test(
            df_eval,
            detect_dbscan_anomalies
        )

        hdbscan_mean, hdbscan_scores = stability_test(
            df_eval,
            detect_hdbscan_anomalies
        )

        st.markdown("### Stability Results ")
        st.write(f"**DBSCAN stability:** {dbscan_mean:.3f}")
        st.write(f"**HDBSCAN stability:** {hdbscan_mean:.3f}")

        

        fig, ax = plt.subplots()
        ax.boxplot(
            [dbscan_scores, hdbscan_scores],
            labels=["DBSCAN", "HDBSCAN"]
        )
        ax.set_ylabel("Jaccard Similarity")
        ax.set_title("Anomaly Stability Under Data Perturbation")
        st.pyplot(fig)



    
    
    elif plot_option == "Monthly Fisheries Outlier Detection":

        import plotly.express as px
       
        # Use your existing prepared monthly data
        #merged_monthly = prepare_monthly(df_land, df_vess)
        # Refresh cached monthly summary if new data was uploaded
        if st.session_state.get("data_updated", False):
            st.cache_data.clear()
            st.session_state.data_updated = False

        merged_monthly = get_monthly_data(df_land, df_vess)

        monthly_outliers = run_monthly_hdbscan_outlier_detection(merged_monthly)

        if monthly_outliers.empty:
            st.warning("Not enough monthly data for HDBSCAN outlier detection.")
            st.stop()

        # VIEW 1 — MONTH-TO-MONTH ANOMALY COUNT
        summary = (
            monthly_outliers
            .groupby("YearMonth")["Anomaly"]
            .sum()
            .reset_index()
            .rename(columns={"Anomaly": "Anomaly Count"})
        )

        fig = px.line(
            summary,
            x="YearMonth",
            y="Anomaly Count",
            markers=True,
            title="Monthly HDBSCAN Anomaly Trend"
        )

        fig.update_layout(
            template="plotly_white",
            xaxis_title="Month",
            yaxis_title="Number of Anomalies"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # VIEW 2 — SELECT A MONTH 
      
        selected_month = st.selectbox(
            "Select Month to Inspect:",
            sorted(monthly_outliers["YearMonth"].unique())
        )

        df_plot = monthly_outliers[
            monthly_outliers["YearMonth"] == selected_month
        ]

        fig = px.scatter(
            df_plot,
            x="Landing",
            y="Vessels",
            title=f"HDBSCAN Monthly Outliers — {selected_month}",
        )

        #  Make NORMAL points solid blue
        fig.update_traces(
            marker=dict(
                color="#1f77b4",   # Plotly default blue (strong)
                size=10,
                opacity=0.7,
                line=dict(width=0)
            ),
            selector=dict(mode="markers")
        )


        anomalies = df_plot[df_plot["Anomaly"]]
        # ==========================================================
        # EXPLAIN WHY EACH MONTHLY ANOMALY WAS FLAGGED
        # ==========================================================
        avg_land = df_plot["Landing"].mean()
        avg_ves  = df_plot["Vessels"].mean()

        def explain_monthly(row):
            if row["Landing"] > avg_land and row["Vessels"] < avg_ves:
                return "⚠️ High landing with relatively few vessels"
            if row["Landing"] < avg_land and row["Vessels"] > avg_ves:
                return "🐟 Low catch efficiency (many vessels, low landing)"
            if row["Landing"] < avg_land and row["Vessels"] < avg_ves:
                return "🛶 Low activity relative to other states"
            if row["Landing"] > avg_land and row["Vessels"] > avg_ves:
                return "🚢 High activity outlier (scale-driven)"
            return "Atypical monthly pattern"

        anomalies = anomalies.copy()
        anomalies["Why Flagged"] = anomalies.apply(explain_monthly, axis=1)


        fig.add_scatter(
            x=anomalies["Landing"],
            y=anomalies["Vessels"],
            mode="markers",
            marker=dict(
                size=14,
                color="red",
                line=dict(width=1, color="black")
            ),
            name="Anomaly",
            hovertext=anomalies["State"]
        )

        fig.update_layout(
            template="plotly_white",
            xaxis_title="Fish Landing (Tonnes)",
            yaxis_title="Number of Fishing Vessels",
            legend_title_text="Legend"
        )

        st.plotly_chart(fig, use_container_width=True)

      
        #  VIEW 3 — TABLE 
       
        st.markdown("### 🚨 Detected Monthly Anomalies")

        st.dataframe(
            anomalies[[
                "Year",
                "Month",
                "State",
                "Landing",
                "Vessels","Why Flagged"
            ]].sort_values(["Year", "Month"]),
            use_container_width=True
        )


    
                    
    elif plot_option == "Fisheries-Based Malaysian States Grouping":
           
        with st.container():
            st.subheader("Malaysian States Fisheries Clustering")
            hierarchical_clustering(merged_df)

           
    

    
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


    elif plot_option == "Geospatial Maps":
        import folium
        import numpy as np
       
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

      
        # EFFICIENCY HEATMAP — FIXED 
        
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
