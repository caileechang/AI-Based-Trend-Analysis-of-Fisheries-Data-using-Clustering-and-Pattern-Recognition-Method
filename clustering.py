# clustering_method.py
# --------------------------------------------------------
# Helper functions for clustering visualizations in Streamlit
# --------------------------------------------------------

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster


# --------------------------------------------------------
# 🔹 Hierarchical Clustering
# --------------------------------------------------------
def hierarchical_clustering(merged_df):
    """Perform hierarchical clustering and show Streamlit visualizations."""
    st.subheader("Hierarchical Clustering (by Year)")

    # Step 1: Scale yearly features
    features = merged_df[['Total Fish Landing (Tonnes)', 'Total number of fishing vessels']]
    scaled_features = StandardScaler().fit_transform(features)

    # Step 2: Compute linkage matrix
    linked = linkage(scaled_features, method='ward')

    # Step 3: Plot dendrogram
    labels = merged_df['Year'].astype(str).tolist()
    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(linked, labels=labels, orientation='top', distance_sort='descending', show_leaf_counts=True, ax=ax)
    ax.set_title("Hierarchical Clustering Dendrogram (by Year)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Distance (Euclidean)")
    st.pyplot(fig)

    # Step 4: Assign cluster labels (e.g., 3 clusters)
    cluster_labels = fcluster(linked, t=3, criterion='maxclust')
    merged_df['Hierarchical_Label'] = cluster_labels

    # Step 5: Scatter plot
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=merged_df,
        x='Total Fish Landing (Tonnes)',
        y='Total number of fishing vessels',
        hue='Hierarchical_Label',
        palette='Set2',
        ax=ax2
    )
    ax2.set_title("Hierarchical Clustering Results (Scatter Plot)")
    st.pyplot(fig2)

    # Step 6: Line trend
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    sns.lineplot(
        data=merged_df,
        x='Year',
        y='Total Fish Landing (Tonnes)',
        hue='Hierarchical_Label',
        marker='o',
        ax=ax3
    )
    ax3.set_title("Fish Landing Trend by Hierarchical Cluster")
    st.pyplot(fig3)

    # Step 7: Clustermap (optional)
    try:
        sns.clustermap(
            merged_df[['Total Fish Landing (Tonnes)', 'Total number of fishing vessels']],
            method='ward',
            cmap='viridis',
            standard_scale=1
        )
        st.pyplot(plt)
    except Exception as e:
        st.warning(f"Unable to display clustermap: {e}")

    # Step 8: Silhouette Score
    try:
        sil_score = silhouette_score(scaled_features, cluster_labels)
        st.success(f"Silhouette Score: {sil_score:.2f}")
    except Exception:
        st.warning("Silhouette score could not be computed (possibly only one cluster).")


# --------------------------------------------------------
# 🔹 DBSCAN Clustering & Anomaly Detection
# --------------------------------------------------------
def dbscan_analysis(merged_df):
    """Perform DBSCAN clustering and visualize anomalies."""
    st.subheader("DBSCAN Clustering & Anomaly Detection")

    # Step 1: Prepare and scale features
    features = merged_df[['Total Fish Landing (Tonnes)', 'Total number of fishing vessels']]
    scaled_features = StandardScaler().fit_transform(features)

    # Step 2: Estimate epsilon (eps) automatically using k-distance
    neighbors = NearestNeighbors(n_neighbors=5)
    distances, _ = neighbors.fit(scaled_features).kneighbors(scaled_features)
    distances = np.sort(distances[:, 4])  # 5th nearest neighbor

    # Step 3: Plot k-distance curve
    fig_k, ax_k = plt.subplots(figsize=(8, 5))
    ax_k.plot(distances)
    ax_k.set_title("k-distance Graph (Elbow for ε selection)")
    ax_k.set_xlabel("Data Points (sorted)")
    ax_k.set_ylabel("5th Nearest Neighbor Distance")
    st.pyplot(fig_k)

    eps_auto = np.percentile(distances, 90)
    st.info(f"Automatically estimated ε (epsilon): `{eps_auto:.3f}`")

    # Step 4: Apply DBSCAN
    db = DBSCAN(eps=eps_auto, min_samples=5)
    labels = db.fit_predict(scaled_features)
    merged_df['DBSCAN_Label'] = labels

    # Step 5: Plot DBSCAN clusters
    fig_db, ax_db = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        x=scaled_features[:, 0],
        y=scaled_features[:, 1],
        hue=labels,
        palette="tab10",
        ax=ax_db
    )
    ax_db.set_title(f"DBSCAN Clustering Results (ε={eps_auto:.3f})")
    ax_db.set_xlabel("Feature 1 (Scaled)")
    ax_db.set_ylabel("Feature 2 (Scaled)")
    st.pyplot(fig_db)

    # Step 6: Compute silhouette score (if valid)
    unique_labels = set(labels) - {-1}
    if len(unique_labels) > 1:
        sil = silhouette_score(scaled_features[labels != -1], labels[labels != -1])
        st.success(f"Silhouette Score (excluding noise): {sil:.3f}")
    else:
        st.warning("Silhouette score not available — only one cluster or all points are noise.")

    # Step 7: Detect anomalies
    anomalies = merged_df[merged_df['DBSCAN_Label'] == -1]
    st.markdown(f"**Detected {len(anomalies)} anomalies (noise points):**")
    if not anomalies.empty:
        st.dataframe(anomalies[['State', 'Year', 'Total Fish Landing (Tonnes)', 'Total number of fishing vessels']])

    # Step 8: Explanation for outliers
    if not anomalies.empty:
        avg_land = merged_df['Total Fish Landing (Tonnes)'].mean()
        avg_vess = merged_df['Total number of fishing vessels'].mean()

        def explain(row):
            if row['Total Fish Landing (Tonnes)'] > avg_land and row['Total number of fishing vessels'] < avg_vess:
                return "High landing but few vessels — possible overperformance or anomaly."
            elif row['Total Fish Landing (Tonnes)'] < avg_land and row['Total number of fishing vessels'] > avg_vess:
                return "Low catch per vessel — possible overfishing or decline."
            elif row['Total Fish Landing (Tonnes)'] < avg_land and row['Total number of fishing vessels'] < avg_vess:
                return "Low overall activity — small fleet or seasonal downtime."
            elif row['Total Fish Landing (Tonnes)'] > avg_land and row['Total number of fishing vessels'] > avg_vess:
                return "Unusually large operation — potential large-scale fishing."
            else:
                return "Atypical pattern compared to average."

        anomalies['Why Flagged'] = anomalies.apply(explain, axis=1)
        st.markdown("### Outlier Explanations")
        st.dataframe(anomalies[['State', 'Year', 'Total Fish Landing (Tonnes)',
                                'Total number of fishing vessels', 'Why Flagged']])
