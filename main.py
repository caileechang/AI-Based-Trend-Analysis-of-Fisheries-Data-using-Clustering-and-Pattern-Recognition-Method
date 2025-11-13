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
from clustering import prepare_monthly, monthly_trends_by_cluster

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
  
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

    st.subheader("Hierarchical Clustering (Total Fish Landing)")

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


   

    # --- Step 2: Let user choose the year ---
    available_years = sorted(df["Year"].dropna().unique())
    selected_year = st.selectbox("Select Year to Cluster:", available_years, index=len(available_years) - 1)

    df_year = df[df["Year"] == selected_year]
    if df_year.empty:
        st.warning("No data available for the selected year.")
        return

    # --- Step 3: Aggregate by state for that year ---
    grouped = (
        df_year.groupby("State")[["Total Fish Landing (Tonnes)", "Total number of fishing vessels"]]
        .sum()
        .reset_index()
    )

    features = ["Total Fish Landing (Tonnes)", "Total number of fishing vessels"]

    # Scale data
    scaled = StandardScaler().fit_transform(grouped[["Total Fish Landing (Tonnes)"]])

    # --- STEP 5: Let user choose linkage method ---
    method = st.selectbox("Select linkage method:", ["ward", "complete", "average", "single"], index=0)

    # --- STEP 6: Compute linkage ---
    linked = linkage(scaled, method=method)

    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(
        linked,
        labels=grouped["State"].tolist(),
        leaf_rotation=45,
        leaf_font_size=9
    )
    ax.set_title(f"Hierarchical Clustering of States – {selected_year} ({method.title()} linkage)")
    ax.set_xlabel("State")
    ax.set_ylabel("Distance")
    st.pyplot(fig)

    # --- Step 7: Optional: allow user to cut dendrogram into clusters ---
    if st.checkbox("Show cluster grouping", value=False):
        num_clusters = st.slider("Select number of clusters", 2, 10, 3)
        grouped["Cluster"] = fcluster(linked, num_clusters, criterion="maxclust")

        st.markdown(f"### Cluster Assignments for {selected_year} (k = {num_clusters})")
        st.dataframe(grouped.sort_values("Cluster").reset_index(drop=True))

        # --- Summary ---
        summary = (
            grouped.groupby("Cluster")[features]
            .mean()
            .reset_index()
        )
        st.markdown(f"### Average Values per Cluster ({selected_year})")
        st.dataframe(summary)
        


    
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
    merged_monthly = prepare_monthly(df_land, df_vess)


    
    st.sidebar.header("Select Visualization")
    plot_option = st.sidebar.radio("Choose a visualization:", [
        "Monthly Trends by Cluster",
        "Yearly Fish Landing Summary",
        "Yearly Cluster Trends for Marine and Freshwater Fish","Optimal K for Monthly & Yearly",                  
        "2D KMeans Scatter",
        "3D KMeans Clustering",
        "Automatic DBSCAN",
        "Hierarchical Clustering",
        "Geospatial Map",
        "Interactive Geospatial Map"
    ])

    
   
   
   
  
   
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
    elif plot_option == "Optimal K for Monthly & Yearly":
        st.subheader("Automatic Determination of Optimal K (Freshwater + Marine Composition)")
    
        # --- Monthly Composition ---
        st.markdown("###  Monthly Fish Landing Composition (Freshwater + Marine)")
    
        # Prepare monthly totals by summing over states for each month
        monthly_comp = (
            df_land.groupby(['Year', 'Month', 'Type of Fish'])['Fish Landing (Tonnes)']
            .sum()
            .reset_index()
            .pivot_table(index=['Year', 'Month'], columns='Type of Fish', values='Fish Landing (Tonnes)', aggfunc='sum')
            .fillna(0)
            .reset_index()
        )
    
        # Rename columns for clarity
        monthly_comp.columns.name = None
        monthly_comp.rename(columns={'Freshwater': 'Freshwater (Tonnes)', 'Marine': 'Marine (Tonnes)'}, inplace=True)
    
        # Scale based on Freshwater & Marine values
        scaled_monthly = StandardScaler().fit_transform(
            monthly_comp[['Freshwater (Tonnes)', 'Marine (Tonnes)']]
        )
    
        best_k_monthly, best_sil_monthly, best_inertia_monthly = evaluate_kmeans_k(
            scaled_monthly, "Monthly Fish Landing (Freshwater + Marine Composition)", use_streamlit=True
        )
    
        # --- Yearly Composition ---
        st.markdown("###  Yearly Fish Landing Composition (Freshwater + Marine)")
    
        yearly_comp = (
            df_land.groupby(['Year', 'Type of Fish'])['Fish Landing (Tonnes)']
            .sum()
            .reset_index()
            .pivot_table(index='Year', columns='Type of Fish', values='Fish Landing (Tonnes)', aggfunc='sum')
            .fillna(0)
            .reset_index()
        )
    
        yearly_comp.columns.name = None
        yearly_comp.rename(columns={'Freshwater': 'Freshwater (Tonnes)', 'Marine': 'Marine (Tonnes)'}, inplace=True)
    
        scaled_yearly = StandardScaler().fit_transform(
            yearly_comp[['Freshwater (Tonnes)', 'Marine (Tonnes)']]
        )
    
        best_k_yearly, best_sil_yearly, best_inertia_yearly = evaluate_kmeans_k(
            scaled_yearly, "Yearly Fish Landing (Freshwater + Marine Composition)", use_streamlit=True
        )
    
        # --- 🧾 Summary ---
        st.markdown("### 🧾 Summary of Optimal K Results (Composition-Based)")
        summary = pd.DataFrame({
            "Dataset": ["Monthly (Freshwater + Marine)", "Yearly (Freshwater + Marine)"],
            "Best K": [best_k_monthly, best_k_yearly],
            "Silhouette Score": [f"{best_sil_monthly:.3f}", f"{best_sil_yearly:.3f}"]
        })
        st.table(summary)
    
        # Store for reuse
        st.session_state['best_k_monthly'] = best_k_monthly
        st.session_state['best_k_yearly'] = best_k_yearly

    elif plot_option == "Yearly Cluster Trends for Marine and Freshwater Fish":
        st.subheader("Yearly K-Means Cluster Trends for Marine and Freshwater Fish")

        # --- User options ---
        period_choice = st.radio("Select period:", ["Yearly", "Monthly"], horizontal=True)
        trend_option = st.radio(
            "Select trend to display:",
            ("Freshwater", "Marine", "Both"),
            horizontal=True
        )
        
        card_style = """
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #444;
        margin-bottom: 15px;
        """
        # ======================================================================================
        #                               YEARLY VIEW
        # ======================================================================================
        if period_choice == "Yearly":
    
            # ------------------------------
            # Aggregate yearly data
            # ------------------------------
            yearly = (
                df_land.groupby(["Year", "Type of Fish"])["Fish Landing (Tonnes)"]
                .sum()
                .reset_index()
                .pivot(index="Year", columns="Type of Fish", values="Fish Landing (Tonnes)")
                .fillna(0)
                .reset_index()
            )
    
            yearly.rename(columns={
                "Freshwater": "Freshwater (Tonnes)",
                "Marine": "Marine (Tonnes)"
            }, inplace=True)
    
            # ------------------------------
            # Clustering
            # ------------------------------
            features = ["Freshwater (Tonnes)", "Marine (Tonnes)"]
            scaled = StandardScaler().fit_transform(yearly[features])
            best_k = st.session_state.get("best_k_yearly", 3)
    
            yearly["Cluster"] = KMeans(n_clusters=best_k, random_state=42).fit_predict(scaled)
    
            st.markdown(f"**Optimal clusters used:** {best_k}")
    
            # Melt for plotting
            melted = yearly.melt(
                id_vars=["Year", "Cluster"],
                value_vars=["Freshwater (Tonnes)", "Marine (Tonnes)"],
                var_name="Type", value_name="Landing"
            )
    
            # ------------------------------
            # Chart styles
            # ------------------------------
            colors = {"Freshwater (Tonnes)": "tab:blue",
                      "Marine (Tonnes)": "tab:red"}
            markers = {"Freshwater (Tonnes)": "o",
                       "Marine (Tonnes)": "^"}
            linestyles = ["solid", "dashed", "dotted", "dashdot"]
    
            # ------------------------------
            # Create line chart
            # ------------------------------
            fig, ax = plt.subplots(figsize=(14, 6))
    
            for fish_type in ["Freshwater (Tonnes)", "Marine (Tonnes)"]:
    
                show_this = (trend_option == "Both" or
                             trend_option.lower() in fish_type.lower())
    
                if show_this:
                    for cl in sorted(melted["Cluster"].unique()):
                        subset = melted[
                            (melted["Type"] == fish_type) &
                            (melted["Cluster"] == cl)
                        ]
    
                        sns.lineplot(
                            data=subset,
                            x="Year", y="Landing",
                            color=colors[fish_type],
                            linestyle=linestyles[cl % len(linestyles)],
                            marker=markers[fish_type],
                            ax=ax,
                            label=f"{fish_type.replace('(Tonnes)','')} – Cluster {cl}"
                        )
    
            ax.set_title(f"Yearly Fish Landing Trends (k={best_k})")
            ax.set_ylabel("Landing (Tonnes)")
            ax.grid(True, alpha=0.3)
    
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)
            st.pyplot(fig)
            
    
            # ------------------------------
            # Summary Cards
            # ------------------------------
            latest_year = yearly["Year"].max()
            prev_year = latest_year - 1
    
            st.markdown(f"## Landing Summary in {latest_year}")
    
            col1, col2 = st.columns(2)
    
            def calc_growth(curr, prev):
                if prev is None or prev == 0:
                    return "–"
                return f"↑ {curr/prev:.2f}x" if curr >= prev else f"↓ {curr/prev:.2f}x"

            def growth_html(curr, prev):
                if prev is None or prev == 0:
                    return "<span style='color:gray;'>–</span>"

                ratio = curr / prev
            
                if ratio >= 1:
                    return f"<span style='color:lightgreen; font-size:20px;'>↑ {ratio:.2f}x</span>"
                else:
                    return f"<span style='color:#ff4d4d; font-size:20px;'>↓ {ratio:.2f}x</span>"


            def safe_get_value(df, year, column):
                row = df.loc[df["Year"] == year, column]
                return row.values[0] if len(row) else 0

    
            if trend_option in ("Freshwater", "Both"):
                fw = safe_get_value(yearly, latest_year, "Freshwater (Tonnes)")
                fw_prev = safe_get_value(yearly, prev_year, "Freshwater (Tonnes)")
            
                with col1:
                    st.markdown(
                        f"""
                        <div style='{card_style}'>
                            <h3 style='color:white;'>Freshwater Landing</h3>
                            <h1 style='color:white; font-size:42px;'><b>{fw:,.0f}</b> tonnes</h1>
                            {growth_html(fw, fw_prev)}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            
            # Marine
            if trend_option in ("Marine", "Both"):
                ma = safe_get_value(yearly, latest_year, "Marine (Tonnes)")
                ma_prev = safe_get_value(yearly, prev_year, "Marine (Tonnes)")
            
                with col2:
                    st.markdown(
                        f"""
                        <div style='{card_style}'>
                            <h3 style='color:white;'>Marine Landing</h3>
                            <h1 style='color:white; font-size:42px;'><b>{ma:,.0f}</b> tonnes</h1>
                            {growth_html(ma, ma_prev)}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
           
        # ======================================================================================
        #                               MONTHLY VIEW
        # ======================================================================================
        else:
    
            # Monthly aggregation
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
    
            # Clustering
            features = ["Freshwater (Tonnes)", "Marine (Tonnes)"]
            scaled = StandardScaler().fit_transform(monthly[features])
            best_k_m = st.session_state.get("best_k_monthly", 3)
    
            monthly["Cluster"] = KMeans(n_clusters=best_k_m, random_state=42).fit_predict(scaled)
    
            st.markdown(f"**Optimal clusters used:** {best_k_m}")
    
            melted = monthly.melt(
                id_vars=["MonthYear", "Cluster"],
                value_vars=["Freshwater (Tonnes)", "Marine (Tonnes)"],
                var_name="Type", value_name="Landing"
            )
    
            # Chart styles
            colors = {"Freshwater (Tonnes)": "tab:blue",
                      "Marine (Tonnes)": "tab:red"}
            markers = {"Freshwater (Tonnes)": "o",
                       "Marine (Tonnes)": "^"}
            linestyles = ["solid", "dashed", "dotted", "dashdot"]
    
            # Plot
            fig, ax = plt.subplots(figsize=(14, 6))
    
            for fish_type in ["Freshwater (Tonnes)", "Marine (Tonnes)"]:
    
                show_this = (trend_option == "Both" or
                             trend_option.lower() in fish_type.lower())
    
                if show_this:
                    for cl in sorted(melted["Cluster"].unique()):
                        subset = melted[
                            (melted["Type"] == fish_type) &
                            (melted["Cluster"] == cl)
                        ]
    
                        sns.lineplot(
                            data=subset,
                            x="MonthYear", y="Landing",
                            color=colors[fish_type],
                            linestyle=linestyles[cl % len(linestyles)],
                            marker=markers[fish_type],
                            ax=ax,
                            label=f"{fish_type.replace('(Tonnes)','')} – Cluster {cl}"
                        )
    
            plt.xticks(rotation=45)
            ax.set_title(f"Monthly Fish Landing Trends (k={best_k_m})")
            ax.set_ylabel("Landing (Tonnes)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)
    
            st.pyplot(fig)
    
            # ===========================
# MONTHLY SUMMARY (Safe Version)
# ===========================
          

            latest_date = monthly["MonthYear"].max()
            prev_date = latest_date - pd.DateOffset(months=1)
            
            st.markdown(f"## Landing Summary in {latest_date.strftime('%B %Y')}")
            
            col1, col2 = st.columns(2)
            
           
                    
            def safe_month_value(df, date, column):
                v = df.loc[df["MonthYear"] == date, column]
                return v.values[0] if len(v) else 0
            
            def calc_growth_month_html(curr, prev):
                """Return colored HTML growth text."""
                if prev is None or prev == 0 or curr==0:
                    return "<span style='color:gray'>–</span>"
                ratio = curr / prev
                if ratio >= 1:
                    return f"<span style='color:green'>↑ {ratio:.2f}x</span>"
                else:
                    return f"<span style='color:red'>↓ {ratio:.2f}x</span>"
            
            # -------- Freshwater Summary --------
            if trend_option in ("Freshwater", "Both"):
                with col1:
                    fw = safe_month_value(monthly, latest_date, "Freshwater (Tonnes)")
                    fw_prev = safe_month_value(monthly, prev_date, "Freshwater (Tonnes)")
            
                    st.markdown(
                        f"""
                        <div style="{card_style}">
                            <h3 style="color:white;">Freshwater Landing</h3>
                            <h1 style="color:white; font-size:42px;"><b>{ma:,.0f}</b> tonnes</h1>
                            {calc_growth_month_html(ma, ma_prev)}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            # -------- Marine Summary --------
            if trend_option in ("Marine", "Both"):
                with col2:
                    ma = safe_month_value(monthly, latest_date, "Marine (Tonnes)")
                    ma_prev = safe_month_value(monthly, prev_date, "Marine (Tonnes)")
            
                    st.markdown(
                        f"""
                        <div style="{card_style}">
                            <h3 style="color:white;">Marine Landing</h3>
                            <h1 style="color:white; font-size:42px;"><b>{ma:,.0f}</b> tonnes</h1>
                            {calc_growth_month_html(ma, ma_prev)}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


    
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
    
        # --- Step 4: User selects visualization mode ---
        vis_mode = st.radio(
            "Select visualization type:",
            ["Static (Matplotlib)", "Interactive (Plotly)"],
            horizontal=True
        )
    
        # --- Shared cluster summary info ---
        st.markdown(f"**Optimal number of clusters:** {best_k}")
        st.markdown("Clusters are automatically determined using the highest silhouette score.")
    
        # --- STATIC (Matplotlib) VERSION -------------------------------------
        if vis_mode == "Static (Matplotlib)":
            st.sidebar.markdown("### Adjust 3D View")
            elev = st.sidebar.slider("Vertical tilt", 0, 90, 30)
            azim = st.sidebar.slider("Horizontal rotation", 0, 360, 45)
    
            fig = plt.figure(figsize=(6, 4))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(
                merged_df['Total number of fishing vessels'],
                merged_df['Total Fish Landing (Tonnes)'],
                merged_df['Year'],
                c=merged_df['Cluster'],
                cmap='viridis',
                s=35, alpha=0.7, edgecolors='k'
            )
    
            ax.tick_params(labelsize=7)
            ax.set_xlabel('Vessels', fontsize=8)
            ax.set_ylabel('Landings', fontsize=8)
            ax.set_zlabel('Year', fontsize=8)
            ax.set_title(f'3D KMeans Clustering (k={best_k})', fontsize=9, pad=5)
            ax.view_init(elev=elev, azim=azim)
            plt.tight_layout(pad=1.0)
            st.pyplot(fig, use_container_width=False)
    
        # --- INTERACTIVE (Plotly) VERSION -------------------------------------
        else:
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
        from scipy.spatial import ConvexHull 

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


        # --- Step 10: Improved cluster visualization ---
          # --- Step 10: Improved cluster visualization ---
        fig, ax = plt.subplots(figsize=(10, 6))
        palette = sns.color_palette("bright", len(unique_labels) + 1)
        n_clusters = len(unique_labels)
    
        for label in np.unique(labels):
            cluster_points = scaled[labels == label]
    
            if label == -1:
                ax.scatter(cluster_points[:, 1], cluster_points[:, 0],
                           s=50, c="lightgray", edgecolor="k", alpha=0.6, label="Noise (-1)")
            else:
                color = palette[label % len(palette)]
                ax.scatter(cluster_points[:, 1], cluster_points[:, 0],
                           s=60, c=[color], edgecolor="k", alpha=0.85,
                           label=f"Cluster {label} ({len(cluster_points)})")
    
                # Draw convex hull around each cluster
                if len(cluster_points) >= 3:
                    hull = ConvexHull(cluster_points)
                    hull_vertices = np.append(hull.vertices, hull.vertices[0])
                    ax.plot(cluster_points[hull_vertices, 1],
                            cluster_points[hull_vertices, 0],
                            color=color, linewidth=2, alpha=0.6)
    
        ax.set_title(f"Automatic DBSCAN (ε={eps_auto:.3f}, min_samples={min_samples_auto}) → {n_clusters} Clusters Found")
        ax.set_xlabel("Vessels (scaled)")
        ax.set_ylabel("Landings (scaled)")
        ax.legend(frameon=True)
        ax.grid(alpha=0.3)
        st.pyplot(fig)
    
        # --- Step 11: Cluster summary table ---
        cluster_summary = merged_df[labels != -1].groupby("DBSCAN_Label")[[
            "Total Fish Landing (Tonnes)", "Total number of fishing vessels"
        ]].mean().reset_index().rename(columns={
            "DBSCAN_Label": "Cluster",
            "Total Fish Landing (Tonnes)": "Avg Fish Landing (Tonnes)",
            "Total number of fishing vessels": "Avg Vessels"
        })
        st.markdown("### 📊 Cluster Summary (excluding noise)")
        st.dataframe(cluster_summary)
    
        # --- Step 12: Outlier analysis ---
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
                    return "⚠️ High landing but few vessels – overperformance or anomaly."
                if r["Total Fish Landing (Tonnes)"] < avg_land and r["Total number of fishing vessels"] > avg_ves:
                    return "🐟 Low catch per vessel – possible overfishing or resource decline."
                if r["Total Fish Landing (Tonnes)"] < avg_land and r["Total number of fishing vessels"] < avg_ves:
                    return "🛶 Low activity – small fleet or seasonal downtime."
                if r["Total Fish Landing (Tonnes)"] > avg_land and r["Total number of fishing vessels"] > avg_ves:
                    return "⚓ Unusually high scale – large operations or exceptional yield."
                return "Atypical pattern vs national average."
    
            outlier_details["Why Flagged"] = outlier_details.apply(explain, axis=1)
            st.markdown("### 🚨 Outlier Details")
            st.dataframe(outlier_details)
    
            # Optional: visual heatmap of outliers
            st.markdown("#### Outlier Heatmap (Catch vs Vessels)")
            fig_h, ax_h = plt.subplots(figsize=(8, 4))
            sns.heatmap(outlier_details[["Total Fish Landing (Tonnes)", "Total number of fishing vessels"]],
                        annot=True, fmt=".0f", cmap="coolwarm", cbar=False, ax=ax_h)
            ax_h.set_title("Outlier Catch-Vessel Patterns")
            st.pyplot(fig_h)
                
    elif plot_option == "Hierarchical Clustering":
                    
        st.subheader("Hierarchical Clustering (by Valid State – Total Fish Landing)")
        
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
            # Melaka
                "MELAKA": [2.1896, 102.2501],
            # Negeri Sembilan
                "NEGERI SEMBILAN": [2.7258, 101.9424],
            # Selangor
                "SELANGOR": [3.0738, 101.5183],
            # Pahang
                "PAHANG": [3.8126, 103.3256],
                "TERENGGANU": [5.3302, 103.1408],
                "KELANTAN": [6.1254, 102.2381],
                "PERAK": [4.5921, 101.0901],
            # Pulau Pinang
                "PULAU PINANG": [5.4164, 100.3327],
            # Kedah
                "KEDAH": [6.1184, 100.3685],
            # Perlis
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

           
if __name__ == "__main__":
    main()
