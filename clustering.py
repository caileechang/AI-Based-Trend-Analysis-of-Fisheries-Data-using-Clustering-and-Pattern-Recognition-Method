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
def monthly_trends_by_cluster(merged_df):
    st.subheader("📅 Monthly Trends by Cluster")
    if merged_df.empty:
        st.warning("No data available.")
        return
    
    scaler = StandardScaler()
    features = merged_df[['Total Fish Landing (Tonnes)', 'Total number of fishing vessels']]
    scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=3, random_state=42)
    merged_df['Cluster'] = kmeans.fit_predict(scaled)

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=merged_df, x='Year', y='Total Fish Landing (Tonnes)', hue='Cluster', marker='o')
    plt.title("Monthly Fish Landing Trends by Cluster")
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
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

    st.subheader("🏗️ Hierarchical Clustering (Grouped by State)")

    # --- STEP 1: Aggregate to reduce label clutter ---
    grouped = (
        merged_df.groupby("State")[["Total Fish Landing (Tonnes)", "Total number of fishing vessels"]]
        .mean()
        .reset_index()
    )

    # --- STEP 2: Scale features ---
    scaler = StandardScaler()
    scaled = scaler.fit_transform(grouped[["Total Fish Landing (Tonnes)", "Total number of fishing vessels"]])

    # --- STEP 3: Compute linkage ---
    linked = linkage(scaled, method='ward')

    # --- STEP 4: Dendrogram with cleaner labels ---
    plt.figure(figsize=(12, 6))
    dendrogram(
        linked,
        labels=grouped["State"].tolist(),
        leaf_rotation=45,
        leaf_font_size=9,
        color_threshold=15
    )
    plt.title("Hierarchical Clustering Dendrogram (by State)")
    plt.xlabel("State")
    plt.ylabel("Distance (Ward linkage)")
    plt.grid(False)
    st.pyplot(plt.gcf())

    # --- STEP 5: Optional cluster assignment ---
    cluster_labels = fcluster(linked, t=3, criterion='maxclust')
    grouped["Cluster"] = cluster_labels

    # --- STEP 6: Show summary ---
    st.dataframe(grouped[["State", "Cluster", "Total Fish Landing (Tonnes)", "Total number of fishing vessels"]])


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


