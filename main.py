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
    land = df_land.copy()

    # --- Standardize State names ---
    land['State'] = (
        land['State'].astype(str).str.upper()
        .str.replace(r'\s*/\s*', '/', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )
    aliases = {
        'JOHOR/JOHORE': 'JOHOR',
        'MELAKA/MALACCA': 'MELAKA',
        'PULAU PINANG/PENANG': 'PULAU PINANG'
    }
    land['State'] = land['State'].replace(aliases)
    land = land[
        ~land['State'].isin(['', 'NAN', 'MALAYSIA:SEMENANJUNG MALAYSIA(PENINSULAR MALAYSIA)', 'JUMLAH'])
    ]

    # --- Normalize column names ---
    land.columns = [c.strip().title() for c in land.columns]

    # --- Normalize “Type Of Fish” values ---
    land['Type Of Fish'] = land['Type Of Fish'].str.strip().str.title()
    land['Type Of Fish'] = land['Type Of Fish'].replace({
        'Fresh Water': 'Freshwater',
        'Fresh Water Fish': 'Freshwater',
        'Marine Fish': 'Marine',
        'Sea Fish': 'Marine'
    })

    # --- Group and pivot ---
    grouped = (
        land.groupby(['Year', 'State', 'Type Of Fish'])['Fish Landing (Tonnes)']
        .sum()
        .reset_index()
    )

    pivot = (
        grouped.pivot_table(
            index=['State', 'Year'],
            columns='Type Of Fish',
            values='Fish Landing (Tonnes)',
            aggfunc='sum'
        )
        .reset_index()
        .fillna(0)
    )

    pivot.rename(
        columns={'Freshwater': 'Freshwater (Tonnes)', 'Marine': 'Marine (Tonnes)'},
        inplace=True
    )

    # --- Merge with vessels ---
    df_vess['State'] = df_vess['State'].astype(str).str.upper().str.strip()
    merged = pd.merge(
        pivot,
        df_vess[['State', 'Year', 'Total number of fishing vessels']],
        on=['State', 'Year'],
        how='outer'
    )

    # --- Fill missing ---
    merged['Freshwater (Tonnes)'] = merged.get('Freshwater (Tonnes)', 0).fillna(0)
    merged['Marine (Tonnes)'] = merged.get('Marine (Tonnes)', 0).fillna(0)
    merged['Total Fish Landing (Tonnes)'] = (
        merged['Freshwater (Tonnes)'] + merged['Marine (Tonnes)']
    )
    merged['Total number of fishing vessels'] = merged['Total number of fishing vessels'].fillna(0)

    return merged

def main():
    st.set_page_config(layout='wide')
    st.title("Fisheries Clustering & Pattern Recognition Dashboard")

    #df_land, df_vess = load_data()
     # --- Load base data only once ---
    if "base_land" not in st.session_state:
        st.session_state.base_land, st.session_state.base_vess = load_data()

    df_land = st.session_state.base_land.copy()
    df_vess = st.session_state.base_vess.copy()


    # Upload additional yearly CSV
    st.sidebar.markdown("### Upload Your Yearly Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload Excel file only (.xlsx)", type=["xlsx"])
  
    

    if uploaded_file:
        try:
            # Try to read both sheets safely
            excel_data = pd.ExcelFile(uploaded_file)
            sheet_names = [s.lower() for s in excel_data.sheet_names]
    
            # Check for both expected sheets
            if "fish landing" in sheet_names and "fish vessels" in sheet_names:
                user_land = pd.read_excel(excel_data, sheet_name="Fish Landing")
                user_vess = pd.read_excel(excel_data, sheet_name="Fish Vessels")
            else:
                st.warning("The uploaded file must contain sheets named 'Fish Landing' and 'Fish Vessels'.")
                user_land, user_vess = None, None
    
            if user_land is not None:
                st.subheader("New dataset uploaded")
                st.dataframe(user_land, use_container_width=True, height=400)
    
                # --- Clean uploaded data ---
                user_land.columns = user_land.columns.str.strip().str.title()
                user_land['Month'] = user_land['Month'].astype(str).str.strip().str.title()
    
                # Convert month names to numbers
                month_map = {
                    'January': 1, 'Jan': 1, 'February': 2, 'Feb': 2, 'March': 3, 'Mar': 3,
                    'April': 4, 'Apr': 4, 'May': 5, 'June': 6, 'Jun': 6, 'July': 7, 'Jul': 7,
                    'August': 8, 'Aug': 8, 'September': 9, 'Sep': 9, 'October': 10, 'Oct': 10,
                    'November': 11, 'Nov': 11, 'December': 12, 'Dec': 12
                }
                user_land['Month'] = user_land['Month'].map(month_map).fillna(user_land['Month'])
                user_land['Month'] = pd.to_numeric(user_land['Month'], errors='coerce')
                user_land['Year'] = pd.to_numeric(user_land['Year'], errors='coerce')
                user_land['Fish Landing (Tonnes)'] = (
                    user_land['Fish Landing (Tonnes)']
                    .astype(str)
                    .str.replace(r'[^\d.]', '', regex=True)
                    .replace('', np.nan)
                    .astype(float)
                )
                user_land.dropna(subset=['Month', 'Year', 'Fish Landing (Tonnes)'], inplace=True)

              

                # Merge with base data
                df_land = pd.concat([df_land, user_land], ignore_index=True).drop_duplicates(subset=['State', 'Year', 'Month'])
                df_vess = pd.concat([df_vess, user_vess], ignore_index=True).drop_duplicates(subset=['State', 'Year'])

                
                # Combine vessel data (from upload if available)
                df_vess = pd.concat([df_vess, user_vess], ignore_index=True).drop_duplicates(subset=['State', 'Year'])
                st.success("Uploaded data successfully merged with existing dataset.")
                st.info(f"Detected uploaded years: {sorted(user_land['Year'].dropna().unique().astype(int).tolist())}")
    
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
    

   
 
    merged_df = prepare_yearly(df_land, df_vess)

 

    
    st.sidebar.header("Select Visualization")
    plot_option = st.sidebar.radio("Choose a visualization:", [
        "Monthly Trends by Cluster",
        "Yearly Fish Landing Summary",
        "Yearly K-Means Cluster Trends",
        "Yearly Elbow & Silhouette",
        "2D KMeans Scatter",
        "3D KMeans Clustering",
        "DBSCAN Anomaly Detection",
         "Automatic DBSCAN",
        "Nested Relationship",
        "Geospatial Map"
    ])

   
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
        st.subheader("Total Yearly Fish Landing by State")

        # If user uploaded a new dataset, re-prepare merged_df dynamically
        if uploaded_file:
            merged_df = prepare_yearly(df_land, df_vess)

        # --- Summarize yearly totals ---
        yearly_summary = (
            merged_df.groupby(['Year', 'State'])[
                ['Freshwater (Tonnes)', 'Marine (Tonnes)', 'Total Fish Landing (Tonnes)']
            ]
            .sum()
            .reset_index()
            .sort_values(['Year', 'State'])
        )
        #yearly_summary = merged_df.groupby(['Year','State'])[['Freshwater (Tonnes)', 'Marine (Tonnes)', 'Total Fish Landing (Tonnes)']].sum().reset_index()
        st.dataframe(yearly_summary, use_container_width=True, height=400)

       
    # Dynamically include newly uploaded years in dropdown
        available_years = sorted([int(y) for y in yearly_summary['Year'].unique()])
        selected_year = st.selectbox("Select a year to view state-level details:", available_years, index=len(available_years) - 1)
    
        # --- Filter and show selected year ---
        filtered = yearly_summary[yearly_summary['Year'] == selected_year]
        st.write(f"### Fish Landing by State for {selected_year}")
        st.dataframe(filtered, use_container_width=True, height=300)
    
        # --- Visualize as bar chart ---
        filtered_sorted = filtered.sort_values('Total Fish Landing (Tonnes)', ascending=False)
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.barplot(
            data=filtered_sorted,
            x='State',
            y='Total Fish Landing (Tonnes)',
            order=filtered_sorted['State'],
            palette='Blues_d',
            ax=ax
        )
    
        # Labels & design
        ax.set_title(f"Total Fish Landing by State - {selected_year}", fontsize=14, pad=15)
        ax.set_xlabel("State", fontsize=12)
        ax.set_ylabel("Total Fish Landing (Tonnes)", fontsize=12)
        plt.xticks(rotation=45, ha='center')
        plt.tight_layout()
    
        # Display bar chart
        st.pyplot(fig)
    
        # Debugging (optional): check available years
        st.sidebar.write("📅 Years currently in dataset:", available_years)
       

    # Allow filtering by year
        #selected_year = st.selectbox("Select a year to view state-level details:", sorted(yearly_summary['Year'].unique()))
       # filtered = yearly_summary[yearly_summary['Year'] == selected_year]
        
        #st.dataframe(filtered, use_container_width=True, height=300)

    

     
# Sort states by total landing for better visual clarity
        #filtered_sorted = filtered.sort_values('Total Fish Landing (Tonnes)', ascending=False)

# Make the figure a bit wider to prevent label overlap
       # fig, ax = plt.subplots(figsize=(14, 6))

# Plot the bars




    elif plot_option == "Yearly K-Means Cluster Trends":
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
        eps = st.sidebar.slider("DBSCAN eps", 0.1, 1.0, 0.1, 0.05)
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


      
    elif plot_option == "Automatic DBSCAN":
        st.subheader("Automatic DBSCAN Clustering & Outlier Detection")

        from sklearn.neighbors import NearestNeighbors
        from kneed import KneeLocator

        # --- Step 1: Select features ---
        features = merged_df[['Total Fish Landing (Tonnes)', 'Total number of fishing vessels']]
        scaled = StandardScaler().fit_transform(features)

        # --- Step 2: Automatically choose min_samples ---
        n_features = scaled.shape[1]
        min_samples_auto = max(3, int(np.log(len(scaled))) + n_features)

        # --- Step 3: Compute k-distances for knee detection ---
        neigh = NearestNeighbors(n_neighbors=min_samples_auto)
        nbrs = neigh.fit(scaled)
        distances, indices = nbrs.kneighbors(scaled)
        distances = np.sort(distances[:, min_samples_auto - 1])  # k-distance

        # --- Step 4: Detect knee point (best epsilon) ---
        kneedle = KneeLocator(range(len(distances)), distances, curve='convex', direction='increasing')
        eps_auto = distances[kneedle.knee] if kneedle.knee else np.percentile(distances, 90)

        st.markdown(f"**Automatically estimated ε (epsilon):** `{eps_auto:.3f}`")
        st.markdown(f"**Automatically chosen min_samples:** `{min_samples_auto}`")

        # --- Step 5: Run DBSCAN ---
        db = DBSCAN(eps=eps_auto, min_samples=min_samples_auto)
        labels = db.fit_predict(scaled)
        merged_df['DBSCAN_Label'] = labels

        # --- Step 6: Visualize clustering ---
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            x=scaled[:, 1],
            y=scaled[:, 0],
            hue=labels,
            palette='tab10',
            s=70,
            ax=ax
        )
        ax.set_title(f"Automatic DBSCAN (ε={eps_auto:.3f}, min_samples={min_samples_auto})")
        ax.set_xlabel("Vessels (scaled)")
        ax.set_ylabel("Landings (scaled)")
        st.pyplot(fig)

        # --- Step 7: Identify and display outliers ---
        n_outliers = (labels == -1).sum()
        st.success(f"Detected {n_outliers} outliers (noise points)")

        if n_outliers > 0:
            outlier_details = merged_df[merged_df['DBSCAN_Label'] == -1][
                ['State', 'Year', 'Total Fish Landing (Tonnes)', 'Total number of fishing vessels']
            ]

            # --- Step 8: Automatically generate 'Why Flagged' explanation ---
            avg_landing = merged_df['Total Fish Landing (Tonnes)'].mean()
            avg_vessels = merged_df['Total number of fishing vessels'].mean()

            def explain_outlier(row):
                if row['Total Fish Landing (Tonnes)'] > avg_landing and row['Total number of fishing vessels'] < avg_vessels:
                    return "High landing but few vessels — possible overperformance or data anomaly."
                elif row['Total Fish Landing (Tonnes)'] < avg_landing and row['Total number of fishing vessels'] > avg_vessels:
                    return "Low catch per vessel — possible overfishing or resource decline."
                elif row['Total Fish Landing (Tonnes)'] < avg_landing and row['Total number of fishing vessels'] < avg_vessels:
                    return "Low overall activity — possibly small fleet or seasonal downtime."
                elif row['Total Fish Landing (Tonnes)'] > avg_landing and row['Total number of fishing vessels'] > avg_vessels:
                    return "Unusually high scale — large operations or exceptional yield."
                else:
                    return "Atypical pattern compared to national average."

            outlier_details['Why Flagged'] = outlier_details.apply(explain_outlier, axis=1)
            st.markdown("### Outlier Details")
            st.dataframe(outlier_details)


    elif plot_option == "Nested Relationship":
            st.subheader("Nested Relationship between Fish Landing, Vessels & States")

            # Example: Boxplot of fish landing by state
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.boxplot(data=merged_df, x='State', y='Total Fish Landing (Tonnes)', hue='Year', ax=ax)
            ax.set_title("Distribution of Fish Landing by State and Year")
            ax.set_xlabel("State")
            ax.set_ylabel("Total Fish Landing (Tonnes)")
            plt.xticks(rotation=45)
            st.pyplot(fig)


    elif plot_option == "Geospatial Map":
        st.subheader("Geospatial Distribution of Fish Landings by Year and Region")

    # Let user choose year
        available_years = sorted(merged_df['Year'].unique())
        selected_year = st.selectbox("Select Year", available_years, index=len(available_years)-1)

    # Filter dataset by selected year
        geo_df = merged_df[merged_df['Year'] == selected_year].copy()

    # Manually define coordinates for each region (including subregions)
        state_coords = {
        # Johor regions
            "JOHOR TIMUR/EAST JOHORE": [2.0, 104.1],
            "JOHOR BARAT/WEST JOHORE": [1.9, 103.3],
            "JOHOR": [1.4854, 103.7618],
        # Melaka
            "MELAKA": [2.1896, 102.2501],
        # Negeri Sembilan
            "NEGERI SEMBILAN": [2.7258, 101.9424],
        # Selangor
            "SELANGOR": [3.0738, 101.5183],
        # Pahang
            "PAHANG": [3.8126, 103.3256],
        # Terengganu
            "TERENGGANU": [5.3302, 103.1408],
        # Kelantan
            "KELANTAN": [6.1254, 102.2381],
        # Perak
            "PERAK": [4.5921, 101.0901],
        # Pulau Pinang
            "PULAU PINANG": [5.4164, 100.3327],
        # Kedah
            "KEDAH": [6.1184, 100.3685],
        # Perlis
            "PERLIS": [6.4449, 100.2048],
        # Sabah & Sarawak regions
            "SABAH": [5.9788, 116.0753],
        
            "SARAWAK": [1.5533, 110.3592],
        # Labuan
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


        
        # Clean coordinate dictionary keys the same way
        clean_coords = {
            re.sub(r'\s*/\s*', '/', k.upper().strip()): v
            for k, v in state_coords.items()
        }
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

        # Optionally add more nested or multi-variable plots here

   
if __name__ == "__main__":
    main()


