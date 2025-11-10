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
from clustering import (monthly_trends_by_cluster,yearly_summary,yearly_kmeans_trends,kmeans_2d,kmeans_3d,hierarchical_clustering,dbscan_analysis)





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
    import numpy as np
    import pandas as pd
    from difflib import get_close_matches

    valid_states = [
        "JOHOR TIMUR/EAST JOHORE", "JOHOR BARAT/WEST JOHORE", "JOHOR",
        "MELAKA", "NEGERI SEMBILAN", "SELANGOR", "PAHANG", "TERENGGANU",
        "KELANTAN", "PERAK", "PULAU PINANG", "KEDAH", "PERLIS",
        "SABAH", "SARAWAK", "W.P. LABUAN"
    ]

    # --- Clean and standardize base dataframe ---
    land = df_land.copy()
    land['State'] = (
        land['State']
        .astype(str)
        .str.upper()
        .str.replace(r'\s*/\s*', '/', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )

    # --- Fuzzy match state names ---
    def match_state(name):
        #if not isinstance(name, str) or name.strip() == "":
            #return np.nan
        matches = get_close_matches(name.upper(), valid_states, n=1, cutoff=0.75)
        return matches[0] if matches else np.nan

    land['State'] = land['State'].apply(match_state)
    land = land[land['State'].isin(valid_states)]

    # --- Group & Pivot directly by Type of Fish ---
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

    # --- Add Total column (always works even if Marine or Freshwater missing) ---
    yearly_pivot['Total Fish Landing (Tonnes)'] = (
        yearly_pivot.get('Freshwater (Tonnes)', 0)
        + yearly_pivot.get('Marine (Tonnes)', 0)
    )

    # --- Vessel totals ---
    for col in ['Inboard Powered', 'Outboard Powered', 'Non-Powered']:
        df_vess[col] = pd.to_numeric(df_vess[col], errors='coerce').fillna(0)
    df_vess['Total number of fishing vessels'] = (
        df_vess['Inboard Powered']
        + df_vess['Outboard Powered']
        + df_vess['Non-Powered']
    )
    df_vess['State'] = df_vess['State'].str.upper().str.strip()
    #df_vess['Year'] = df_vess['Year'].astype(int)
    # Convert to numeric safely first
    df_vess['Year'] = pd.to_numeric(df_vess['Year'], errors='coerce')
    # Drop rows where Year is missing (NaN) to avoid casting error
    df_vess = df_vess.dropna(subset=['Year'])
    df_vess['Year'] = df_vess['Year'].astype(int)


    # --- Merge fish landing with vessel data ---
    merged = pd.merge(
        yearly_pivot,
        df_vess[['State', 'Year', 'Total number of fishing vessels']],
        on=['State', 'Year'],
        how='outer'   # full outer join — keep all states
    ).fillna(0)


    return merged.sort_values(['Year', 'State']).reset_index(drop=True)


    
def main():
    st.set_page_config(layout='wide')
    st.title("Fisheries Clustering & Pattern Recognition Dashboard")

    #df_land, df_vess = load_data()
    


    # --- Load base data or use newly merged uploaded data ---
    if "base_land" not in st.session_state:
        st.session_state.base_land, st.session_state.base_vess = load_data()
        st.session_state.data_updated = False  # no uploaded data yet
    
    # If a new dataset has been uploaded previously, use that merged version
    if "data_updated" in st.session_state and st.session_state.data_updated:
        df_land = st.session_state.base_land.copy()
        df_vess = st.session_state.base_vess.copy()
    else:
        # otherwise, use the original base data
        df_land = st.session_state.base_land.copy()
        df_vess = st.session_state.base_vess.copy()


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


                  
        
                    # --- Now analyze both datasets together ---
                   # merged_df = prepare_yearly(df_land, df_vess)
        
                   # st.write("Merged Yearly Fish Landing Data")
                   # st.dataframe(merged_df, use_container_width=True, height=300)
        
                  #  years = sorted(merged_df['Year'].unique())
                    #selected_year = st.selectbox("Select a year to view state-level details:", years, index=len(years)-1)
                 #   filtered_year = merged_df[merged_df['Year'] == selected_year]
        
                   # st.subheader(f"Fish Landing by State for {selected_year}")
                   # st.dataframe(filtered_year, use_container_width=True)
                    #st.bar_chart(filtered_year, x="State", y="Total Fish Landing (Tonnes)", use_container_width=True)
        
            except Exception as e:
                st.error(f"Error reading uploaded file: {e}")

    merged_df = prepare_yearly(df_land, df_vess)


    
    st.sidebar.header("Select Visualization")
    plot_option = st.sidebar.radio("Choose a visualization:", [
        "Monthly Trends by Cluster","Monthly Trends by Cluster2",
        "Yearly Fish Landing Summary","Yearly Summary",
        "Yearly K-Means Cluster Trends","Yearly KMeans Trends",
        #"Yearly Elbow & Silhouette",
        "2D KMeans Scatter","2D KMeans Scatter2",
        "3D KMeans Clustering","3D KMeans Clustering2",
        "3--D KMeans Clustering","DBSCAN Clustering",
        #"DBSCAN Anomaly Detection",
        "Automatic DBSCAN",
        "Hierarchical Clustering", "Hierarchical Clustering2", "DBSCAN Analysis",
        "Geospatial Map",
        "Interactive Geospatial Map"
    ])

    if plot_option == "Monthly Trends by Cluster2":
        monthly_trends_by_cluster(merged_df)
    elif plot_option == "Yearly Summary":
        yearly_summary(merged_df)
    elif plot_option == "Yearly KMeans Trends":
        yearly_kmeans_trends(merged_df)
    elif plot_option == "2D KMeans Scatter2":
        kmeans_2d(merged_df)
    elif plot_option == "3D KMeans Clustering2":
        kmeans_3d(merged_df)
    elif plot_option == "Hierarchical Clustering2":
        hierarchical_clustering(merged_df)
    elif plot_option == "DBSCAN Analysis":
        dbscan_analysis(merged_df)
   
    if plot_option == "Monthly Trends by Cluster":
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
        # Ensure numeric month and valid values only
        #monthly['Month'] = pd.to_numeric(monthly['Month'], errors='coerce')
        #monthly = monthly.dropna(subset=['Year', 'Month'])
        
        # Convert to first day of month safely
       # monthly['MonthYear'] = pd.to_datetime(
         #   monthly['Year'].astype(int).astype(str) + '-' + monthly['Month'].astype(int).astype(str) + '-01',
            #errors='coerce'
        #)
        #monthly = monthly.dropna(subset=['MonthYear'])

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
        #selected_year = st.selectbox("Select a year to view state-level details:", available_years, index=len(available_years) - 1)
        selected_year = st.selectbox(
            "Select a year to view state-level details:",
            available_years,
            index=len(available_years) - 1,
            key="yearly_summary_selectbox"
        )

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
        st.subheader("Automatic 3D K-Means Clustering")
    
        from mpl_toolkits.mplot3d import Axes3D
    
        # --- Step 1: Prepare data ---
        features = merged_df[['Total Fish Landing (Tonnes)', 'Total number of fishing vessels']]
        scaled = StandardScaler().fit_transform(features)
    
        # --- Step 2: Automatically find best k (Silhouette) ---
        sil_scores = {}
        for k in range(2, 11):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(scaled)
            sil_scores[k] = silhouette_score(scaled, labels)
    
        best_k = max(sil_scores, key=sil_scores.get)
    
        # --- Step 3: Apply final model ---
        final_model = KMeans(n_clusters=best_k, random_state=42)
        merged_df['Cluster'] = final_model.fit_predict(scaled)

             # --- Step 4: Let user control camera angle ---
        st.sidebar.markdown("### Adjust 3D View")
        elev = st.sidebar.slider("Vertical tilt)", 0, 90, 30)
        azim = st.sidebar.slider("Horizontal rotation)", 0, 360, 45)
    
        # --- Step 4: 3D Plot ---
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(
            merged_df['Total number of fishing vessels'],
            merged_df['Total Fish Landing (Tonnes)'],
            merged_df['Year'],
            c=merged_df['Cluster'],
            cmap='viridis',
            s=35, alpha=0.7,edgecolors='k'
        )
         
        ax.tick_params(labelsize=7)
        ax.set_xlabel('Vessels', fontsize=8)
        ax.set_ylabel('Landings', fontsize=8)
        ax.set_zlabel('Year', fontsize=8)
        ax.set_title(f'3D KMeans Clustering (k={best_k})', fontsize=9, pad=5)
        ax.view_init(elev=elev, azim=azim)
        plt.tight_layout(pad=1.0)
        st.pyplot(fig, use_container_width=False)


    
    elif plot_option == "3--D KMeans Clustering":
        st.subheader("Interactive 3D K-Means Clustering")
    
        # --- Step 1: Prepare data ---
        features = merged_df[['Total Fish Landing (Tonnes)', 'Total number of fishing vessels']]
        scaled = StandardScaler().fit_transform(features)
    
        # --- Step 2: Automatically find best k (Silhouette) ---
        sil_scores = {}
        for k in range(2, 11):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(scaled)
            sil_scores[k] = silhouette_score(scaled, labels)
    
        best_k = max(sil_scores, key=sil_scores.get)
    
        # --- Step 3: Apply final model ---
        final_model = KMeans(n_clusters=best_k, random_state=42)
        merged_df['Cluster'] = final_model.fit_predict(scaled)
    
        # --- Step 4: Create interactive 3D scatter ---
        fig = px.scatter_3d(
            merged_df,
            x='Total number of fishing vessels',
            y='Total Fish Landing (Tonnes)',
            z='Year',
            color='Cluster',
            color_continuous_scale='Viridis',
            title=f"3D KMeans Clustering (k={best_k})",
            height=600
        )
    
        fig.update_traces(marker=dict(size=4, line=dict(width=0.5, color='DarkSlateGrey')))
        fig.update_layout(
            scene=dict(
                xaxis_title="Vessels",
                yaxis_title="Landings",
                zaxis_title="Year"
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
    
        st.plotly_chart(fig, use_container_width=True)
        
           
        
     

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

        # --- Step 1: Keep only valid Malaysian states ---
        valid_states = [
            "JOHOR TIMUR/EAST JOHORE", "JOHOR BARAT/WEST JOHORE", "JOHOR",
            "MELAKA", "NEGERI SEMBILAN", "SELANGOR", "PAHANG", "TERENGGANU",
            "KELANTAN", "PERAK", "PULAU PINANG", "KEDAH", "PERLIS",
            "SABAH", "SARAWAK", "W.P. LABUAN"
        ]
        merged_df = merged_df[merged_df["State"].isin(valid_states)].reset_index(drop=True)
        if merged_df.empty:
            st.warning("No valid data rows remain after state filtering.")
            st.stop()

        # --- Step 2: Select and scale features ---
        features = merged_df[["Total Fish Landing (Tonnes)", "Total number of fishing vessels"]]
        scaled = StandardScaler().fit_transform(features)

        # --- Step 3: Automatically choose min_samples ---
        n_features = scaled.shape[1]
        min_samples_auto = max(3, int(np.log(len(scaled))) + n_features)

        # --- Step 4: Compute k-distances for knee detection ---
        neigh = NearestNeighbors(n_neighbors=min_samples_auto)
        distances, _ = neigh.fit(scaled).kneighbors(scaled)
        distances = np.sort(distances[:, min_samples_auto - 1])

        # --- Step 5: Detect knee point (best epsilon) ---
        kneedle = KneeLocator(range(len(distances)), distances, curve="convex", direction="increasing")
        eps_auto = distances[kneedle.knee] if kneedle.knee is not None else np.percentile(distances, 90)

        # --- Step 6: Display auto parameters ---
        st.markdown(f"**Automatically estimated ε (epsilon):** `{eps_auto:.3f}`")
        st.markdown(f"**Automatically chosen min_samples:** `{min_samples_auto}`")

        # --- Step 7: Plot sorted k-distance graph (justification) ---
        fig_k, ax_k = plt.subplots(figsize=(8, 5))
        ax_k.plot(distances, label="Sorted k-distance")
        if kneedle.knee is not None:
            ax_k.axvline(kneedle.knee, color="red", linestyle="--", label=f"Elbow index = {kneedle.knee}")
            ax_k.axhline(eps_auto, color="green", linestyle="--", label=f"ε = {eps_auto:.3f}")
        ax_k.set_title("Sorted k-distance Graph (Elbow Method for ε Selection)")
        ax_k.set_xlabel("Points sorted by distance")
        ax_k.set_ylabel("k-distance")
        ax_k.legend()
        st.pyplot(fig_k)

        # --- Step 8: Run DBSCAN ---
        db = DBSCAN(eps=eps_auto, min_samples=min_samples_auto)
        labels = db.fit_predict(scaled)
        merged_df["DBSCAN_Label"] = labels

        # --- Step 9: Cluster quality metric (silhouette) ---
       #if len(set(labels)) > 1 and np.any(labels != -1):
          #  sil = silhouette_score(scaled[labels != -1], labels[labels != -1])
         #   st.info(f"Silhouette Score (excluding noise): `{sil:.3f}`")
       # else:
            #st.warning("Silhouette score unavailable (all points are noise or only one cluster).")
        # --- Step 9: Cluster quality metric (silhouette) ---
        unique_labels = set(labels) - {-1}  # remove noise label (-1)
        if len(unique_labels) > 1:
            try:
                sil = silhouette_score(scaled[labels != -1], labels[labels != -1])
                st.info(f"Silhouette Score (excluding noise): `{sil:.3f}`")
            except Exception as e:
                st.warning(f"Silhouette score could not be computed: {e}")
        else:
            st.warning("Silhouette score unavailable — only one cluster or all points labeled as noise.")


        # --- Step 10: Visualize clustering results ---
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=scaled[:, 1], y=scaled[:, 0], hue=labels, palette="tab10", s=70, ax=ax)
        ax.set_title(f"Automatic DBSCAN (ε={eps_auto:.3f}, min_samples={min_samples_auto})")
        ax.set_xlabel("Vessels (scaled)")
        ax.set_ylabel("Landings (scaled)")
        st.pyplot(fig)

        # --- Step 11: Identify and explain outliers ---
        n_outliers = (labels == -1).sum()
        st.success(f"Detected {n_outliers} outliers (noise points)")

        if n_outliers > 0:
            outlier_details = merged_df.loc[labels == -1, [
                "State", "Year", "Total Fish Landing (Tonnes)", "Total number of fishing vessels"
            ]].copy()

            avg_land = merged_df["Total Fish Landing (Tonnes)"].mean()
            avg_ves = merged_df["Total number of fishing vessels"].mean()

            def explain(r):
                if r["Total Fish Landing (Tonnes)"] > avg_land and r["Total number of fishing vessels"] < avg_ves:
                    return "High landing but few vessels – possible overperformance or data anomaly."
                if r["Total Fish Landing (Tonnes)"] < avg_land and r["Total number of fishing vessels"] > avg_ves:
                    return "Low catch per vessel – possible overfishing or resource decline."
                if r["Total Fish Landing (Tonnes)"] < avg_land and r["Total number of fishing vessels"] < avg_ves:
                    return "Low overall activity – small fleet or seasonal downtime."
                if r["Total Fish Landing (Tonnes)"] > avg_land and r["Total number of fishing vessels"] > avg_ves:
                    return "Unusually high scale – large operations or exceptional yield."
                return "Atypical pattern compared to national average."

            outlier_details["Why Flagged"] = outlier_details.apply(explain, axis=1)
            st.markdown("### Outlier Details")
            st.dataframe(outlier_details)

   
       
        elif plot_option == "Hierarchical Clustering":
            hierarchical_clustering(merged_df)

        elif plot_option == "DBSCAN Clustering":
            dbscan_analysis(merged_df)
        



 
            

    

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

            m = folium.Map(location=[4.5, 109.5], zoom_start=6, tiles=None)
            #folium.TileLayer("CartoDB positron", name=None, control=False).add_to(m)
            folium.TileLayer("OpenStreetMap", name="Default Map").add_to(m)
            folium.TileLayer("CartoDB positron", name="Light Mode",
                            attr="© OpenStreetMap contributors © CARTO").add_to(m)
            
                

                
                # --- Step 6: Add Color Scale ---
            min_val = float(geo_df['Total Fish Landing (Tonnes)'].min())
            max_val = float(geo_df['Total Fish Landing (Tonnes)'].max())
                    
            colormap = cm.LinearColormap(
                    colors=['blue', 'lime', 'yellow', 'orange', 'red'],
                    vmin=min_val, vmax=max_val, caption = f"Fish Landing (Tonnes)\nMin: {min_val:,.0f}  |  Max: {max_val:,.0f}"
                )
                

    # Force legend to show exact min & max numbers
            
            colormap.add_to(m)
            
        

            # colormap = cm.linear.YlGnBu_09.scale(min_val, max_val)
            # colormap.caption = "Fish Landing (Tonnes)"
            # colormap.add_to(m)

                # --- Step 7: Add Circle Markers (clickable points) ---
            for _, row in geo_df.iterrows():
                    popup_html = (
                        f"<b>{row['State']}</b><br>"
                        f"Fish Landing: {row['Total Fish Landing (Tonnes)']:.2f} tonnes<br>"
                        f"Fish Vessels: {row['Total number of fishing vessels']:.0f}"
                    )

                    color = colormap(row['Total Fish Landing (Tonnes)'])

                    folium.CircleMarker(
                        location=row['Coords'],
                        radius=10,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.9,
                        popup=folium.Popup(popup_html, max_width=250),
                        tooltip=row['State']
                    ).add_to(m)
                    

            geo_df['HeatValue']=np.log1p(geo_df['Total Fish Landing (Tonnes)'])
            
                # --- Step 8: Heatmap Layer ---
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

            min_val = geo_df['Total Fish Landing (Tonnes)'].min()
            max_val = geo_df['Total Fish Landing (Tonnes)'].max()
                
            HeatMap(heat_data, name="Fish Landing Heatmap", radius=15, blur=8, min_opacity=0.5,max_opacity=0.95,gradient=gradient,max_val=max_val).add_to(m)

            
                # --- Step 9: Map Controls ---
            MiniMap(toggle_display=True).add_to(m)
            Fullscreen(position='topright').add_to(m)
            folium.LayerControl(collapsed=False).add_to(m)
        
                # --- Step 10: Display Map ---
            st_folium(m, use_container_width=True, height=600, returned_objects=[])

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

    

                    

        

   
if __name__ == "__main__":
    main()
