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

def main():
    st.set_page_config(layout='wide')
    st.title("🌊 Fisheries Clustering & Pattern Recognition Dashboard")

    df_land, df_vess = load_data()

    # Upload additional yearly CSV
    st.sidebar.markdown("### 📤 Upload Your Yearly CSV")
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        try:
            user_df = pd.read_csv(uploaded_file)
            st.subheader("📈 User Uploaded Yearly Data Preview")
            st.dataframe(user_df.head())
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")

    st.sidebar.header("Select Visualization")
    plot_option = st.sidebar.radio("Choose a visualization:", [
        "Monthly Trends by Cluster",
        "Yearly Fish Landing Summary",
        "Yearly K-Means Cluster Trends",
        "Yearly Elbow & Silhouette",
        "2D KMeans Scatter",
        "3D KMeans Clustering",
        "DBSCAN Anomaly Detection",
        "Nested Relationship"
    ])

    def prepare_yearly(df_land, df_vess):
        land = df_land.copy()
        land['State'] = (
            land['State'].astype(str).str.upper()
            .str.replace(r'\s*/\s*', '/', regex=True)
            .str.replace(r'\s+', ' ', regex=True)
            .str.strip()
        )
        aliases = {'JOHOR/JOHORE': 'JOHOR', 'MELAKA/MALACCA': 'MELAKA', 'PULAU PINANG/PENANG': 'PULAU PINANG'}
        land['State'] = land['State'].replace(aliases)
        land = land[~land['State'].isin(['', 'NAN', 'MALAYSIA:SEMENANJUNG MALAYSIA(PENINSULAR MALAYSIA)'])]

        grouped = land.groupby(['Year', 'State', 'Type of Fish'])['Fish Landing (Tonnes)'].sum().reset_index()
        pivot = grouped.pivot_table(index=['State', 'Year'], columns='Type of Fish', values='Fish Landing (Tonnes)', aggfunc='sum').reset_index().fillna(0)
        pivot.rename(columns={'Freshwater': 'Freshwater (Tonnes)', 'Marine': 'Marine (Tonnes)'}, inplace=True)

        merged = pd.merge(pivot, df_vess[['State', 'Year', 'Total number of fishing vessels']], on=['State', 'Year'], how='inner')
        merged['Total Fish Landing (Tonnes)'] = merged['Freshwater (Tonnes)'] + merged['Marine (Tonnes)']
        return merged

    merged_df = prepare_yearly(df_land, df_vess)

    if plot_option == "Monthly Trends by Cluster":
        monthly = df_land.groupby(['Year', 'Month'])['Fish Landing (Tonnes)'].sum().reset_index()
        monthly['MonthYear'] = pd.to_datetime(monthly['Year'].astype(str) + '-' + monthly['Month'].astype(str).str.zfill(2))
        X = StandardScaler().fit_transform(monthly[['Month', 'Fish Landing (Tonnes)']])
        monthly['Cluster'] = KMeans(n_clusters=3, random_state=42).fit_predict(X)

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=monthly.sort_values('MonthYear'), x='MonthYear', y='Fish Landing (Tonnes)', hue='Cluster', marker='o', ax=ax)
        ax.set_title("Monthly Fish Landing Trends by Cluster")
        ax.set_xlabel("Month-Year")
        ax.set_ylabel("Fish Landing (Tonnes)")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    elif plot_option == "Yearly Fish Landing Summary":
        st.subheader("📊 Total Yearly Fish Landing (Merged Dataset)")
        yearly_summary = merged_df.groupby('Year')[['Freshwater (Tonnes)', 'Marine (Tonnes)', 'Total Fish Landing (Tonnes)']].sum().reset_index()
        st.dataframe(yearly_summary)

    elif plot_option == "Yearly KMeans Cluster Trends":
        features = merged_df[['Freshwater (Tonnes)', 'Marine (Tonnes)']]
        scaled = StandardScaler().fit_transform(features)
        merged_df['Cluster'] = KMeans(n_clusters=3, random_state=42).fit_predict(scaled)

        cluster_trends = merged_df.groupby(['Year', 'Cluster'])[['Freshwater (Tonnes)', 'Marine (Tonnes)']].mean().reset_index()
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=cluster_trends, x='Year', y='Freshwater (Tonnes)', hue='Cluster', marker='o', ax=ax)
        ax.set_title("Yearly Freshwater Landing Trends by Cluster")
        st.pyplot(fig)

    elif plot_option == "Yearly Elbow & Silhouette":
        features = merged_df[['Total Fish Landing (Tonnes)', 'Total number of fishing vessels']]
        scaled = StandardScaler().fit_transform(features)
        ks = range(2, 11)
        inertia = []
        silhouette = []

        for k in ks:
            model = KMeans(n_clusters=k, random_state=42).fit(scaled)
            inertia.append(model.inertia_)
            silhouette.append(silhouette_score(scaled, model.labels_))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(ks, inertia, marker='o')
        ax1.set_title("Elbow Method")
        ax1.set_xlabel("k")
        ax1.set_ylabel("Inertia")
        ax2.plot(ks, silhouette, marker='o', color='orange')
        ax2.set_title("Silhouette Score")
        ax2.set_xlabel("k")
        ax2.set_ylabel("Score")
        st.pyplot(fig)

    elif plot_option == "2D KMeans Scatter":
        k = st.sidebar.slider("Select k for KMeans", 2, 10, 3)
        features = merged_df[['Total Fish Landing (Tonnes)', 'Total number of fishing vessels']]
        scaled = StandardScaler().fit_transform(features)
        merged_df['Cluster'] = KMeans(n_clusters=k, random_state=42).fit_predict(scaled)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=merged_df, x='Total number of fishing vessels', y='Total Fish Landing (Tonnes)', hue='Cluster', palette='viridis', s=70, ax=ax)
        ax.set_title(f"KMeans Clustering (k={k})")
        st.pyplot(fig)

    elif plot_option == "3D KMeans Clustering":
        from mpl_toolkits.mplot3d import Axes3D
        features = merged_df[['Total Fish Landing (Tonnes)', 'Total number of fishing vessels']]
        scaled = StandardScaler().fit_transform(features)
        merged_df['Cluster'] = KMeans(n_clusters=3, random_state=42).fit_predict(scaled)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(merged_df['Total number of fishing vessels'], merged_df['Total Fish Landing (Tonnes)'], merged_df['Year'], c=merged_df['Cluster'], cmap='viridis', s=60)
        ax.set_xlabel('Vessels')
        ax.set_ylabel('Landings')
        ax.set_zlabel('Year')
        ax.set_title('3D KMeans Clustering')
        st.pyplot(fig)

    elif plot_option == "DBSCAN Anomaly Detection":
        eps = st.sidebar.slider("DBSCAN eps", 0.1, 1.0, 0.5, 0.05)
        features = merged_df[['Total Fish Landing (Tonnes)', 'Total number of fishing vessels']]
        scaled = StandardScaler().fit_transform(features)
        labels = DBSCAN(eps=eps, min_samples=5).fit_predict(scaled)
        merged_df['DBSCAN_Label'] = labels

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=scaled[:, 1], y=scaled[:, 0], hue=labels, palette='tab10', ax=ax)
        ax.set_title(f"DBSCAN Clustering (eps={eps})")
        ax.set_xlabel("Vessels (scaled)")
        ax.set_ylabel("Landings (scaled)")
        st.pyplot(fig)

        st.markdown(f"**Outliers Detected:** {(labels == -1).sum()}")

    elif plot_option == "Nested Relationship":
        st.subheader("🔗 Nested Relationship between Fish Landing, Vessels & States")

        # Example: Boxplot of fish landing by state
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=merged_df, x='State', y='Total Fish Landing (Tonnes)', hue='Year', ax=ax)
        ax.set_title("Distribution of Fish Landing by State and Year")
        ax.set_xlabel("State")
        ax.set_ylabel("Total Fish Landing (Tonnes)")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Optionally add more nested or multi-variable plots here


if __name__ == "__main__":
    main()
