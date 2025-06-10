import streamlit as st
import pandas as pd
import numpy as np
import calendar
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

@st.cache_data
def load_data():
    """
    Load and clean the 'Fish Landing' and 'Fish Vessels' sheets.
    """
    url = 'https://www.dropbox.com/scl/fi/4cl5zaor1l32ikyudvf2e/Fisheries-Dataset-vessels-fish-landing.xlsx?rlkey=q2ewpeuzj288ewd17rcqxeuie&st=6h4zijb8&dl=1'
    # Load sheets
    df_land = pd.read_excel(url, sheet_name='Fish Landing')
    df_vess = pd.read_excel(url, sheet_name='Fish Vessels')

    # --- Clean landing data ---
    # Convert fish landing to numeric
    df_land['Fish Landing (Tonnes)'] = (
        df_land['Fish Landing (Tonnes)']
        .astype(str)
        .str.replace(r'[^\d.]', '', regex=True)
        .replace('', np.nan)
        .astype(float)
    )
    df_land = df_land.dropna(subset=['Fish Landing (Tonnes)']).reset_index(drop=True)
    # Convert month names to numbers
    df_land['Month'] = df_land['Month'].apply(
        lambda x: list(calendar.month_name).index(x.strip().title())
        if isinstance(x, str) else x
    )

    # --- Clean vessel data ---
    for col in ['Inboard Powered', 'Outboard Powered', 'Non-Powered']:
        df_vess[col] = pd.to_numeric(df_vess[col], errors='coerce').fillna(0)
    df_vess['Total number of fishing vessels'] = (
        df_vess['Inboard Powered'] +
        df_vess['Outboard Powered'] +
        df_vess['Non-Powered']
    )
    df_vess['State'] = df_vess['State'].str.upper().str.strip()
    df_vess['Year'] = df_vess['Year'].astype(int)

    return df_land, df_vess


def prepare_merged(df_land, df_vess):
    """
    Merge cleaned landing and vessel data, add clustering.
    """
    # Clean and normalize state names in landing data
    df = df_land.copy()
    df['State'] = (
        df['State'].astype(str)
        .str.upper()
        .str.replace(r'\s*/\s*', '/', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )
    df = df[~df['State'].isin(['', 'NAN'])]
    df = df[df['State'] != 'MALAYSIA:SEMENANJUNG MALAYSIA(PENINSULAR MALAYSIA)']

    # Replace common aliases
    aliases = {
        'JOHOR/JOHORE': 'JOHOR',
        'MELAKA/MALACCA': 'MELAKA',
        'PULAU PINANG/PENANG': 'PULAU PINANG'
    }
    df['State'] = df['State'].replace(aliases)

    # Pivot fish landing
    landing = df.groupby(['Year', 'State', 'Type of Fish'])['Fish Landing (Tonnes)'] \
                .sum().reset_index()
    pivot = landing.pivot_table(
        index=['State', 'Year'],
        columns='Type of Fish',
        values='Fish Landing (Tonnes)',
        aggfunc='sum'
    ).reset_index().fillna(0)
    pivot = pivot.rename(columns={
        'Freshwater': 'Freshwater (Tonnes)',
        'Marine': 'Marine (Tonnes)'
    })

    # Merge with vessel counts
    merged = pd.merge(
        pivot,
        df_vess[['State', 'Year', 'Total number of fishing vessels']],
        on=['State', 'Year'],
        how='inner'
    )
    merged['Total Fish Landing (Tonnes)'] = (
        merged['Freshwater (Tonnes)'] + merged['Marine (Tonnes)']
    )

    # Add a simple k-means cluster
    X = StandardScaler().fit_transform(
        merged[['Total Fish Landing (Tonnes)', 'Total number of fishing vessels']]
    )
    merged['Cluster'] = KMeans(n_clusters=3, random_state=42) \
                        .fit_predict(X)

    return merged


def plot_monthly_trends(df_land):
    """
    Line plot of monthly total fish landings colored by k-means cluster.
    """
    df = (
        df_land.groupby(['Year', 'Month'])['Fish Landing (Tonnes)']
        .sum().reset_index()
        .rename(columns={'Fish Landing (Tonnes)': 'Total Fish Landing (Tonnes)'})
    )
    df['MonthYear'] = pd.to_datetime(
        df['Year'].astype(str) + '-' + df['Month'].astype(str).str.zfill(2)
    )

    X = StandardScaler().fit_transform(
        df[['Month', 'Total Fish Landing (Tonnes)']]
    )
    df['Cluster'] = KMeans(n_clusters=3, random_state=42).fit_predict(X)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(
        data=df.sort_values('MonthYear'),
        x='MonthYear',
        y='Total Fish Landing (Tonnes)',
        hue='Cluster',
        marker='o',
        ax=ax
    )
    ax.set_title('Monthly Fish Landing Trends by Cluster')
    plt.xticks(rotation=45)
    return fig


def plot_yearly_elbow(scaled_features):
    """
    Elbow plot for k-means inertia over a range of k.
    """
    inertias = []
    ks = range(2, 11)
    for k in ks:
        inertias.append(
            KMeans(n_clusters=k, random_state=42)
            .fit(scaled_features).inertia_
        )

    fig, ax = plt.subplots()
    ax.plot(list(ks), inertias, marker='o')
    ax.set_title('Elbow Method: Inertia vs. k')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Inertia (WCSS)')
    return fig


def plot_3d_kmeans(merged_df):
    """
    3D scatter of vessels vs. landings vs. year colored by k-means cluster.
    """
    X = StandardScaler().fit_transform(
        merged_df[['Total number of fishing vessels', 'Total Fish Landing (Tonnes)']]
    )
    labels = merged_df['Cluster']

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        merged_df['Year'],
        X[:, 0],
        X[:, 1],
        c=labels,
        cmap='viridis',
        s=60
    )
    ax.set_xlabel('Year')
    ax.set_ylabel('Vessels (scaled)')
    ax.set_zlabel('Landings (scaled)')
    ax.set_title('3D K-Means Clustering')
    return fig


def plot_dbscan(merged_df, eps):
    """
    2D DBSCAN scatter for landings vs. vessels.
    """
    X = StandardScaler().fit_transform(
        merged_df[['Total Fish Landing (Tonnes)', 'Total number of fishing vessels']]
    )
    labels = DBSCAN(eps=eps, min_samples=5).fit_predict(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        x=X[:, 1],
        y=X[:, 0],
        hue=labels,
        palette='tab10',
        ax=ax
    )
    ax.set_title(f'DBSCAN Clustering (eps={eps})')
    ax.set_xlabel('Vessels (scaled)')
    ax.set_ylabel('Landings (scaled)')
    return fig

def plot_kmeans_scatter(merged_df, k=3):
    """
    2D scatter: total vessels vs total landings, colored by KMeans cluster.
    """
    # 1) Scale your two features
    X = StandardScaler().fit_transform(
        merged_df[['Total number of fishing vessels', 'Total Fish Landing (Tonnes)']]
    )
    # 2) Fit & label
    labels = KMeans(n_clusters=k, random_state=42).fit_predict(X)

    # 3) Build the figure
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        x=merged_df['Total number of fishing vessels'],
        y=merged_df['Total Fish Landing (Tonnes)'],
        hue=labels,
        palette='viridis',
        ax=ax,
        s=60
    )
    ax.set_title(f'K-Means Scatter (k={k})')
    ax.set_xlabel('Total number of vessels')
    ax.set_ylabel('Total fish landings (Tonnes)')
    return fig

def plot_silhouette_elbow(scaled_features, title_prefix="Data"):
    """
    Returns a figure with two subplots:
      • Silhouette score vs k  
      • Inertia vs k
    """
    ks = list(range(2, 11))
    sil_scores = []
    inertias = []

    for k in ks:
        km = KMeans(n_clusters=k, random_state=42).fit(scaled_features)
        sil_scores.append(silhouette_score(scaled_features, km.labels_))
        inertias.append(km.inertia_)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(ks, sil_scores, marker='o')
    ax1.set(title=f'{title_prefix} Silhouette',
            xlabel='k', ylabel='Silhouette Score')
    ax2.plot(ks, inertias, marker='o', color='orange')
    ax2.set(title=f'{title_prefix} Elbow (Inertia)',
            xlabel='k', ylabel='Inertia (WCSS)')
    fig.tight_layout()
    return fig

def main():
    st.set_page_config(layout='wide')
    st.title("🐟 Fisheries Clustering Dashboard")

    df_land, df_vess = load_data()
    merged_df = prepare_merged(df_land, df_vess)

    st.sidebar.header("Choose a plot")
    choice = st.sidebar.radio("", [
        "Monthly Trends by Cluster",
        "Yearly Elbow Method",
        "Silhouette & Elbow",
        "K-Means Scatter",
        "3D K-Means",
        "DBSCAN Anomaly Detection"
    ])

    if choice == "Monthly Trends by Cluster":
        fig = plot_monthly_trends(df_land)
        st.pyplot(fig)

    elif choice == "Yearly Elbow Method":
        X = StandardScaler().fit_transform(
            merged_df[['Total Fish Landing (Tonnes)', 'Total number of fishing vessels']]
        )
        fig = plot_yearly_elbow(X)
        st.pyplot(fig)

    elif choice == "Silhouette & Elbow":
        X = StandardScaler().fit_transform(
            merged_df[['Total Fish Landing (Tonnes)', 'Total number of fishing vessels']]
        )
        fig = plot_silhouette_elbow(X, "Yearly Data")
        st.pyplot(fig)

    elif choice == "K-Means Scatter":
        k = st.sidebar.slider("Select k", 2, 10, 3)
        fig = plot_kmeans_scatter(merged_df, k)
        st.pyplot(fig)

    elif choice == "3D K-Means":
        fig = plot_3d_kmeans(merged_df)
        st.pyplot(fig)

    else:  # DBSCAN
        eps = st.sidebar.slider("DBSCAN eps", 0.1, 1.0, 0.5, 0.05)
        fig = plot_dbscan(merged_df, eps)
        st.pyplot(fig)
        outliers = (DBSCAN(eps=eps, min_samples=5)
                    .fit_predict(StandardScaler().fit_transform(
                        merged_df[['Total Fish Landing (Tonnes)','Total number of fishing vessels']]
                    )) == -1).sum()
        st.write(f"Outliers detected: {outliers}")

if __name__ == "__main__":
    main()
