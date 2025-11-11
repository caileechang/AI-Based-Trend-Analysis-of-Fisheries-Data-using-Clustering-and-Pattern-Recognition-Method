import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# ========================================
# 1️⃣ MONTHLY CLUSTER TRENDS
# ========================================
def prepare_monthly(df_land, df_vess):
    import pandas as pd
    import numpy as np

    # --- Step 1: Basic cleaning ---
    df_land = df_land.copy()
    df_land['Month'] = pd.to_numeric(df_land['Month'], errors='coerce')
    df_land['Year'] = pd.to_numeric(df_land['Year'], errors='coerce')
    df_land['Fish Landing (Tonnes)'] = pd.to_numeric(df_land['Fish Landing (Tonnes)'], errors='coerce')

    # Remove invalid or missing entries
    df_land = df_land.dropna(subset=['Month', 'Year', 'Fish Landing (Tonnes)', 'State'])

    # --- Step 2: Group by Year + Month + State ---
    monthly_totals = (
        df_land.groupby(['Year', 'Month', 'State'], as_index=False)['Fish Landing (Tonnes)']
        .sum()
    )

    # --- Step 3: Clean & aggregate vessel data ---
    for col in ['Inboard Powered', 'Outboard Powered', 'Non-Powered']:
        df_vess[col] = pd.to_numeric(df_vess[col], errors='coerce').fillna(0)

    df_vess['Total number of fishing vessels'] = (
        df_vess['Inboard Powered'] + df_vess['Outboard Powered'] + df_vess['Non-Powered']
    )
    df_vess['State'] = df_vess['State'].astype(str).str.upper().str.strip()
    df_vess['Year'] = pd.to_numeric(df_vess['Year'], errors='coerce')

    # --- Step 4: Merge by State + Year ---
    merged_monthly = pd.merge(
        monthly_totals,
        df_vess[['State', 'Year', 'Total number of fishing vessels']],
        on=['State', 'Year'],
        how='left'
    )

    # --- Step 5: Drop rows with missing months ---
    merged_monthly = merged_monthly.dropna(subset=['Month'])

    # --- Step 6: Sort chronologically ---
    merged_monthly = merged_monthly.sort_values(['Year', 'Month', 'State']).reset_index(drop=True)

    return merged_monthly

def monthly_trends_by_cluster(merged_df):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    import streamlit as st

    st.subheader("📅 Monthly Trends by Cluster")

    # --- Step 1: Ensure Month column exists ---
    if 'Month' not in merged_df.columns:
        st.warning("⚠️ Month data not available in merged_df. Using yearly data instead.")
        return

    # --- Step 2: Aggregate by Year + Month (summed across states) ---
    monthly_data = (
        merged_df.groupby(['Year', 'Month'], as_index=False)[
            ['Total Fish Landing (Tonnes)', 'Total number of fishing vessels']
        ]
        .sum()
    )

    # --- Step 3: Scale and cluster ---
    scaler = StandardScaler()
    scaled = scaler.fit_transform(monthly_data[['Total Fish Landing (Tonnes)', 'Total number of fishing vessels']])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    monthly_data['Cluster'] = kmeans.fit_predict(scaled)

    # --- Step 4: Create combined Month-Year label ---
    monthly_data['MonthYear'] = pd.to_datetime(
        monthly_data['Year'].astype(str) + '-' + monthly_data['Month'].astype(str).str.zfill(2),
        errors='coerce'
    )

    # --- Step 5: Plot trend ---
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=monthly_data,
        x='MonthYear',
        y='Total Fish Landing (Tonnes)',
        hue='Cluster',
        marker='o',
        palette='tab10'
    )
    plt.title("Monthly Fish Landing Trends by Cluster")
    plt.xlabel("Month-Year")
    plt.ylabel("Total Fish Landing (Tonnes)")
    plt.xticks(rotation=45)
    plt.grid(True)
    st.pyplot(plt.gcf())




# ========================================
# 2️⃣ YEARLY SUMMARY
# ========================================
def yearly_summary(merged_df):
    st.subheader("📈 Yearly Fish Landing Summary")
    yearly = merged_df.groupby('Year')['Total Fish Landing (Tonnes)'].sum().reset_index()

    plt.figure(figsize=(12, 5))
    sns.barplot(data=yearly, x='Year', y='Total Fish Landing (Tonnes)', palette='Blues_d')
    plt.title("Yearly Fish Landing Totals")
    plt.grid(True)
    st.pyplot(plt.gcf())

# ========================================
# 3️⃣ YEARLY K-MEANS CLUSTER TRENDS
# ========================================
def yearly_kmeans_trends(merged_df):
    st.subheader("📊 Yearly K-Means Cluster Trends")
    scaler = StandardScaler()
    features = merged_df[['Total Fish Landing (Tonnes)', 'Total number of fishing vessels']]
    scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=5, random_state=42)
    merged_df['Cluster'] = kmeans.fit_predict(scaled)

    yearly_cluster = merged_df.groupby(['Year', 'Cluster'])[['Total Fish Landing (Tonnes)']].mean().reset_index()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=yearly_cluster, x='Year', y='Total Fish Landing (Tonnes)', hue='Cluster', marker='o')
    plt.title("Yearly Cluster Trends (KMeans)")
    plt.grid(True)
    st.pyplot(plt.gcf())

# ========================================
# 4️⃣ 2D K-MEANS SCATTER
# ========================================
def kmeans_2d(merged_df):
    st.subheader("🎯 2D K-Means Clustering")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(merged_df[['Total Fish Landing (Tonnes)', 'Total number of fishing vessels']])
    kmeans = KMeans(n_clusters=3, random_state=42)
    merged_df['Cluster'] = kmeans.fit_predict(scaled)

    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=merged_df,
                    x='Total Fish Landing (Tonnes)',
                    y='Total number of fishing vessels',
                    hue='Cluster',
                    palette='viridis',
                    s=80)
    plt.title("2D KMeans Clustering")
    plt.grid(True)
    st.pyplot(plt.gcf())

# ========================================
# 5️⃣ 3D K-MEANS CLUSTERING
# ========================================
from mpl_toolkits.mplot3d import Axes3D
def kmeans_3d(merged_df):
    st.subheader("🧭 3D K-Means Clustering")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(merged_df[['Total Fish Landing (Tonnes)', 'Total number of fishing vessels']])
    kmeans = KMeans(n_clusters=3, random_state=42)
    merged_df['Cluster'] = kmeans.fit_predict(scaled)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        merged_df['Total number of fishing vessels'],
        merged_df['Total Fish Landing (Tonnes)'],
        merged_df['Year'],
        c=merged_df['Cluster'],
        cmap='viridis',
        s=50
    )
    ax.set_xlabel('Fishing Vessels')
    ax.set_ylabel('Fish Landing (Tonnes)')
    ax.set_zlabel('Year')
    plt.title('3D K-Means Clustering')
    st.pyplot(fig)

# ========================================
# 6️⃣ HIERARCHICAL CLUSTERING
# ========================================
def hierarchical_clustering(merged_df):
    import streamlit as st
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

    st.subheader("Hierarchical Clustering (by Valid State – Total Fish Landing)")

    # --- STEP 1: Define valid Malaysian states ---
    valid_states = [
        "JOHOR TIMUR/EAST JOHORE", "JOHOR BARAT/WEST JOHORE", "JOHOR",
        "MELAKA", "NEGERI SEMBILAN", "SELANGOR", "PAHANG", "TERENGGANU",
        "KELANTAN", "PERAK", "PULAU PINANG", "KEDAH", "PERLIS",
        "SABAH", "SARAWAK", "W.P. LABUAN"
    ]

    # --- STEP 2: Clean data and filter for valid states ---
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
        st.warning("No valid state records found after filtering.")
        return

    # --- STEP 3: Aggregate by state (average total fish landing) ---
    grouped = (
        df.groupby("State")[["Total Fish Landing (Tonnes)"]]
        .mean()
        .reset_index()
    )

    # --- STEP 4: Scale data ---
    scaled = StandardScaler().fit_transform(grouped[["Total Fish Landing (Tonnes)"]])

    # --- STEP 5: Let user choose linkage method ---
    method = st.selectbox("Select linkage method:", ["ward", "complete", "average", "single"], index=0)

    # --- STEP 6: Compute linkage ---
    linked = linkage(scaled, method=method)

    # --- STEP 7: Dendrogram plot ---
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(linked, labels=grouped["State"].tolist(), leaf_rotation=45, leaf_font_size=9)
    ax.set_title(f"Hierarchical Clustering Dendrogram ({method.title()} linkage)")
    ax.set_xlabel("State")
    ax.set_ylabel("Distance")
    st.pyplot(fig)

    # --- STEP 8: Let user dynamically select number of clusters ---
    num_clusters = st.slider("Select number of clusters", 2, 10, 3)
    grouped["Cluster"] = fcluster(linked, num_clusters, criterion="maxclust")

    # --- STEP 9: Display cluster results ---
    st.markdown(f"### Cluster Assignments (k = {num_clusters})")
    st.dataframe(grouped.sort_values("Cluster").reset_index(drop=True))

    # --- STEP 10: Cluster summary ---
    summary = (
        grouped.groupby("Cluster")["Total Fish Landing (Tonnes)"]
        .mean()
        .reset_index()
        .sort_values("Cluster")
    )
    st.markdown("### Average Total Fish Landing per Cluster")
    st.dataframe(summary)

    # --- STEP 11: Optional: CSV download ---
    csv = summary.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Cluster Summary (CSV)",
        csv,
        "hierarchical_total_fish_landing_summary.csv",
        "text/csv"
    )


# ========================================
# 7️⃣ DBSCAN CLUSTERING
# ========================================
def dbscan_analysis(merged_df):
    st.subheader("🔍 DBSCAN Anomaly Detection")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(merged_df[['Total Fish Landing (Tonnes)', 'Total number of fishing vessels']])
    db = DBSCAN(eps=0.45, min_samples=5).fit(scaled)
    merged_df['DBSCAN_Label'] = db.labels_

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=merged_df,
        x='Total Fish Landing (Tonnes)',
        y='Total number of fishing vessels',
        hue='DBSCAN_Label',
        palette='tab10'
    )
    plt.title("DBSCAN Clustering (Outliers = -1)")
    plt.grid(True)
    st.pyplot(plt.gcf())


