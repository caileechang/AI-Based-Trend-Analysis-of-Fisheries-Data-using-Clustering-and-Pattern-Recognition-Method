   elif plot_option == "Unified HDBSCAN ":
        import folium
       
        import seaborn as sns
        from streamlit_folium import st_folium
    
        
        from sklearn.preprocessing import StandardScaler

        st.subheader("Unified HDBSCAN Outlier Detection (Monthly + Yearly)")
        st.markdown(
            "<p style='color:#aaa'>Detect both monthly and yearly anomalies with map & explanations.</p>",
            unsafe_allow_html=True
        )

        # -----------------------------
        # Select YEAR
        # -----------------------------
        years = sorted(merged_df["Year"].unique())
        sel_year = st.selectbox("Select Year:", years)

        # ===========================================================
        #                  CREATE THE 4 TABS
        # ===========================================================
        tab1, tab2, tab3, tab4 = st.tabs([
            "🟦 Yearly Outliers",
            "🟩 Monthly Outliers",
            "🗺️ Map View",
            "📘 Explanation"
        ])

        # ===========================================================
        #                   1️⃣ YEARLY OUTLIER TAB
        # ===========================================================
        with tab1:

            st.markdown("### 🟦 Yearly Outliers Summary")

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
                st.warning("No data available for this year.")
            else:
                # Scaling
                X_year = StandardScaler().fit_transform(df_yearly[["Landing", "Vessels"]])

                # HDBSCAN
                yearly_clusterer = hdbscan.HDBSCAN(
                    min_samples=3, min_cluster_size=3, prediction_data=True
                ).fit(X_year)

                df_yearly["Outlier_Score"] = yearly_clusterer.outlier_scores_

                if df_yearly["Outlier_Score"].max() > 0:
                    df_yearly["Outlier_Norm"] = df_yearly["Outlier_Score"] / df_yearly["Outlier_Score"].max()
                else:
                    df_yearly["Outlier_Norm"] = 0

                df_yearly["Anomaly"] = df_yearly["Outlier_Norm"] >= 0.65

                # Explanation
                avgL = df_yearly["Landing"].mean()
                avgV = df_yearly["Vessels"].mean()

                def exp_y(row):
                    L, V = row["Landing"], row["Vessels"]
                    if L > avgL and V < avgV: return "⚡ High landing, few vessels"
                    if L < avgL and V > avgV: return "🐟 Low catch per vessel"
                    if L < avgL and V < avgV: return "🛶 Low activity"
                    if L > avgL and V > avgV: return "⚓ Intensive fishing"
                    return "Unusual pattern"

                df_yearly["Explanation"] = df_yearly.apply(exp_y, axis=1)

                out_y = df_yearly[df_yearly["Anomaly"] == True][[
                    "State", "Landing", "Vessels", "Outlier_Norm", "Explanation"
                ]]

                if out_y.empty:
                    st.success("No yearly anomalies detected.")
                else:
                    st.dataframe(out_y, use_container_width=True)

                # -------- YEARLY SCATTER PLOT --------
                st.markdown("### 📈 Yearly Landing vs Vessels")
                fig, ax = plt.subplots(figsize=(9, 5))

                sns.scatterplot(
                    data=df_yearly,
                    x="Vessels", y="Landing",
                    hue="Outlier_Norm",
                    palette="viridis", s=120, ax=ax
                )

                # highlight outliers
                ano = df_yearly[df_yearly["Anomaly"] == True]
                ax.scatter(
                    ano["Vessels"], ano["Landing"],
                    facecolors="none", edgecolors="red",
                    s=250, linewidths=2
                )

                for _, r in ano.iterrows():
                    ax.text(r["Vessels"], r["Landing"], r["State"],
                            color="red", fontsize=9, fontweight="bold")

                ax.grid(alpha=0.3)
                st.pyplot(fig)

        # ===========================================================
        #                   2️⃣ MONTHLY OUTLIER TAB
        # ===========================================================
        with tab2:

            st.markdown("### 🟩 Monthly Outliers Summary")

            df_month = merged_monthly[merged_monthly["Year"] == sel_year].copy()
            all_month_outliers = []

            for m in sorted(df_month["Month"].unique()):
                df_m = df_month[df_month["Month"] == m].copy()

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

                # HDBSCAN
                X_m = StandardScaler().fit_transform(df_m[["Landing", "Vessels"]])
                cl_m = hdbscan.HDBSCAN(min_samples=3, min_cluster_size=3,
                                    prediction_data=True).fit(X_m)

                df_m["Outlier_Score"] = cl_m.outlier_scores_

                if df_m["Outlier_Score"].max() > 0:
                    df_m["Outlier_Norm"] = df_m["Outlier_Score"] / df_m["Outlier_Score"].max()
                else:
                    df_m["Outlier_Norm"] = 0

                df_m["Anomaly"] = df_m["Outlier_Norm"] >= 0.65

                # Explanation
                avgL = df_m["Landing"].mean()
                avgV = df_m["Vessels"].mean()

                def exp_m(row):
                    L, V = row["Landing"], row["Vessels"]
                    if L > avgL and V < avgV: return "⚡ High landing, few vessels"
                    if L < avgL and V > avgV: return "🐟 Low catch per vessel"
                    if L < avgL and V < avgV: return "🛶 Low activity"
                    if L > avgL and V > avgV: return "⚓ Intensive fishing"
                    return "Unusual pattern"

                df_m["Explanation"] = df_m.apply(exp_m, axis=1)

                out_m = df_m[df_m["Anomaly"] == True][[
                    "Year", "Month", "State", "Landing", "Vessels",
                    "Outlier_Norm", "Explanation"
                ]]

                if not out_m.empty:
                    all_month_outliers.append(out_m)

            # final combined monthly outliers
            if len(all_month_outliers) == 0:
                st.success("No monthly anomalies detected.")
            else:
                result = pd.concat(all_month_outliers).sort_values(["Month", "State"])
                st.dataframe(result, use_container_width=True)

        # ===========================================================
        #                   3️⃣ MAP VIEW TAB
        # ===========================================================
        with tab3:

            st.markdown("### 🗺️ Malaysia Outlier Map (Yearly + Monthly)")

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

            # Use yearly df from tab1
            df_y = df_yearly.copy()
            df_y["Coords"] = df_y["State"].map(coords)

            m = folium.Map(location=[4.5, 109.5], zoom_start=6)

            for _, row in df_y.iterrows():
                if row["Coords"] is None:
                    continue
                lat, lon = row["Coords"]
                color = "red" if row["Anomaly"] else "blue"
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=8, color=color, fill=True, fill_color=color,
                    tooltip=f"{row['State']} — {'Anomaly' if row['Anomaly'] else 'Normal'}"
                ).add_to(m)

            st_folium(m, height=550, width=800)

        # ===========================================================
        #                   4️⃣ EXPLANATION TAB
        # ===========================================================
        with tab4:
            st.markdown("### 📘 How HDBSCAN Outlier Detection Works")
            st.write("""
            **HDBSCAN** finds natural density clusters and identifies points that do not fit into any cluster 
            (noise/outliers).  

            ### ✔ Why monthly & yearly outliers differ
            - Monthly = short-term anomalies  
            - Yearly = long-term structural changes  
            - Different number of rows → different density  
            - More variation in monthly → more outliers  

            ### ✔ Outlier Threshold (Normalized ≥ 0.65)
            - Values closer to **1.0** = very strong anomaly  
            - Below **0.65** = normal range  

            ### ✔ Interpretation Rules
            - ⚡ High landing + few vessels → efficient or sudden spike  
            - 🐟 Low catch per vessel → possible overfishing  
            - 🛶 Low activity → seasonal or low effort  
            - ⚓ Intensive fishing → large fleets, high volume  
            """)



elif plot_option == "3D KMeans Clustering":
        import matplotlib.pyplot as plt
        import seaborn as sns
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
            ["Static", "Interactive"],
            horizontal=True
        )
    
        # --- Shared cluster summary info ---
        st.markdown(f"**Optimal number of clusters:** {best_k}")
        st.markdown("Clusters are automatically determined using the highest silhouette score.")
    
        # --- STATIC (Matplotlib) VERSION -------------------------------------
        if vis_mode == "Static":
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



             elif plot_option == "Geospatial Map(Heatmap)":
        import folium
        from streamlit_folium import st_folium
        from folium.plugins import HeatMap
        from branca.colormap import linear

        st.subheader("🌍 Interactive Geospatial Heatmap")
        st.markdown("""
        <p style='color:#ccc'>
        Explore Malaysia’s fish landing distribution using an intuitive interactive heatmap.
        Use map themes, search, tooltips, and layer toggles for a better experience.
        </p>
        """, unsafe_allow_html=True)


      
        # ----------------------------------------------------
        # 1. PREPARE CLEAN YEARLY DATA
        # ----------------------------------------------------
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

        df_year = df_year[df_year["Landing"] > 0]

        # ----------------------------------------------------
        # 2. STATE COORDINATES
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
        df = df_year.dropna(subset=["Coords"]).copy()

      
        all_states = sorted(df_year["State"].unique())

        selected_states = st.multiselect(
            "Select State(s):",
            all_states,
            default=all_states  # by default show all states
        )

        # Filter dataframe based on selection
        df = df_year[df_year["State"].isin(selected_states)].copy()

        if df.empty:
            st.warning("No states selected.")
            st.stop()


        

        # ----------------------------------------------------
        # 4. MAP THEME SELECTOR
        # ----------------------------------------------------
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
        # 5. BUILD MAP (AUTO-ZOOM)
        # ----------------------------------------------------
        min_lat = min(df["Coords"].apply(lambda x: x[0]))
        max_lat = max(df["Coords"].apply(lambda x: x[0]))
        min_lon = min(df["Coords"].apply(lambda x: x[1]))
        max_lon = max(df["Coords"].apply(lambda x: x[1]))

        m = folium.Map(
            tiles=None,
            zoom_start=6
        )

        # Add base layer but HIDE from layer control
        folium.TileLayer(
            tile_map[theme],
            name="Base Map",
            control=False
        ).add_to(m)


        m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

        # ----------------------------------------------------
        # 6. COLOR SCALE LEGEND
        # ----------------------------------------------------
        min_v = df["Landing"].min()
        max_v = df["Landing"].max()

        ticks = [
            min_v,
            min_v + (max_v - min_v) * 0.25,
            min_v + (max_v - min_v) * 0.50,
            min_v + (max_v - min_v) * 0.75,
            max_v
        ]

        cmap = linear.Blues_09.scale(min_v, max_v).to_step(5)

        cmap.caption = (
            f"Fish Landing (Tonnes)\n"
            f"Min: {min_v:,.0f}   |   Max: {max_v:,.0f}"
        )

        m.add_child(cmap)
      
  # ----------------------------------------------------
        # 3. SUMMARY CARDS (Top, Bottom, Total)
        # ----------------------------------------------------
        total = df["Landing"].sum()
        highest = df.loc[df["Landing"].idxmax()]
        lowest = df.loc[df["Landing"].idxmin()]

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
        # 7. HEATMAP LAYER
        # ----------------------------------------------------
        heat_data = [
            [row["Coords"][0], row["Coords"][1], row["Landing"]]
            for _, row in df.iterrows()
        ]

        heat_group = folium.FeatureGroup("Heatmap")
        HeatMap(
            heat_data,
            radius=40, blur=25, min_opacity=0.4,
        ).add_to(heat_group)
        heat_group.add_to(m)

        # ----------------------------------------------------
        # 8. MARKERS (CIRCLEMARKERS)
        # ----------------------------------------------------
        marker_group = folium.FeatureGroup("State Markers")

        for _, r in df.iterrows():
            lat, lon = r["Coords"]
            val = r["Landing"]

            tooltip = f"""
            <b>{r['State']}</b><br>
            Landing: {val:,.0f} tonnes<br>
            Vessels: {r['Vessels']:,.0f}
            """

            folium.CircleMarker(
                location=[lat, lon],
                radius=9,
                color="#ffffff",
                fill=True,
                fill_color=cmap(val),
                fill_opacity=0.95,
                weight=1.3,
                tooltip=tooltip
            ).add_to(marker_group)

        marker_group.add_to(m)
        # ----------------------------------------------------
        # 10. DISPLAY MAP
        # ----------------------------------------------------
        st_folium(m, height=550, width="100%")

        # ----------------------------------------------------
        # 11. TABLE BELOW MAP
        # ----------------------------------------------------
        st.markdown("### 📋 State Landing Table")
        st.dataframe(
            df.sort_values("Landing", ascending=False).reset_index(drop=True),
            use_container_width=True,
            height=300
        )

        # ----------------------------------------------------
        # 12. INTERPRETATION
        # ----------------------------------------------------
        with st.expander("ℹ️ How to read this map"):
            st.markdown("""
            **Heatmap intensity** reflects the amount of fish landed:
            - Darker blue → Higher landing  
            - Light blue → Lower landing  
            - Hover circles to view detailed values  
            - Use layer panel to toggle heatmap or markers  
            - Use search box to jump directly to any state  
            """)



            elif plot_option == "Yearly Fish Landing Summary":
        import seaborn as sns
        import matplotlib.pyplot as plt

        st.subheader("Total Yearly Fish Landing by State")
        # --- ALWAYS use cleaned yearly summary from prepare_yearly ---
        if uploaded_file:
            yearly_summary = prepare_yearly(df_land, df_vess)
        else:
            yearly_summary = st.session_state.get("yearly_summary", prepare_yearly(df_land, df_vess))
    
        # Store globally for other visualisations
        st.session_state.yearly_summary = yearly_summary
    
        # --- SHOW CLEAN TABLE ---
        st.dataframe(yearly_summary, use_container_width=True, height=400)

        # --- Year selector ---
        available_years = sorted([int(y) for y in yearly_summary['Year'].unique()])
    
        selected_year = st.selectbox(
            "Select a year to view state-level details:",
            available_years,
            index=len(available_years) - 1,
            key="yearly_summary_selectbox"
        )
    
        # --- Filter for selected year ---
        filtered = yearly_summary[yearly_summary['Year'] == selected_year]
    
        st.write(f"### Fish Landing by State for {selected_year}")
        st.dataframe(filtered, use_container_width=True, height=300)
    
        # --- Sort states for chart ---
        filtered_sorted = filtered.sort_values('Total Fish Landing (Tonnes)', ascending=True)
    
        # --- Horizontal bar chart (Safe size for Streamlit Cloud) ---
        fig, ax = plt.subplots(figsize=(10, 7))
    
        sns.barplot(
            data=filtered_sorted,
            x='Total Fish Landing (Tonnes)',
            y='State',
            palette='Blues_d',
            ax=ax
        )
    
        ax.set_title(f"Total Fish Landing by State - {selected_year}", fontsize=14, pad=12)
        ax.set_xlabel("Total Fish Landing (Tonnes)", fontsize=12)
        ax.set_ylabel("State", fontsize=12)
    
        # Value labels
        for i, v in enumerate(filtered_sorted['Total Fish Landing (Tonnes)']):
            ax.text(
                v + (0.01 * v), i,
                f"{v:,.0f}",
                va='center',
                fontsize=9
            )
    
        plt.tight_layout()
        st.pyplot(fig)




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
    
        # st.markdown("## Fish Landing Trends (Cluster-Based Analysis)")
        st.markdown("""
            <h2 style='color:white;'>🎣 Fish Landing Trends (Cluster Analysis)</h2>
            <p style='color:#ccc; margin-top:-10px;'>
                Compare freshwater & marine fish landings across yearly or monthly periods using K-Means cluster grouping.
            </p>
        """, unsafe_allow_html=True)

        st.markdown("<hr style='border:0.5px solid #444;'>", unsafe_allow_html=True)
        # Options box
        with st.container():
            #st.markdown("<h4 style='color:white;'> Display Options</h4>", unsafe_allow_html=True)
            st.markdown(
                "<p style='color:#ccc; margin-top:-12px; font-size:15px;'>"
                "Please select the period and trend to display the fish landing analysis."
                "</p>", 
                unsafe_allow_html=True
            )

            opt_col1, opt_col2 = st.columns([1,2])

            with opt_col1:
                period_choice = st.radio("Period:", ["Yearly", "Monthly"], horizontal=True)

            with opt_col2:
                trend_option = st.radio("Trend:", ["Freshwater", "Marine", "Both"], horizontal=True)
                
        # period_choice = st.radio("Select period:", ["Yearly", "Monthly"], horizontal=True)
    
        # trend_option = st.radio(
           # "Select trend to display:",
         #   ("Freshwater", "Marine", "Both"),
          #  horizontal=True
        #)
  
        # YEARLY SUMMARY (Shown only if yearly selected)   
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
                if prev == 0:
                    return "<span style='color:gray;'>–</span>"
                ratio = curr / prev
                if ratio >= 1:
                    return f"<span style='color:lightgreen; font-size:20px;'>↑ {ratio:.2f}x</span>"
                else:
                    return f"<span style='color:#ff4d4d; font-size:20px;'>↓ {ratio:.2f}x</span>"
    
            fw_latest = safe_get(yearly, latest_year, "Freshwater (Tonnes)")
            fw_prev = safe_get(yearly, prev_year, "Freshwater (Tonnes)")
            ma_latest = safe_get(yearly, latest_year, "Marine (Tonnes)")
            ma_prev = safe_get(yearly, prev_year, "Marine (Tonnes)")
    
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
   
            # YEARLY CLUSTER PLOT
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
  
        # MONTHLY VIEW
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
    
            # Summary section (monthly)
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
    
            # =============== Monthly Cluster Plot ===============
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

        
    elif plot_option == "2D KMeans Scatter":
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
    
