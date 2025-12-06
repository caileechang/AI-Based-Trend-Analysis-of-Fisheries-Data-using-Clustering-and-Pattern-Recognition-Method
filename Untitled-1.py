elif plot_option == "Optimal K for Monthly & Yearly":

        import pandas as pd

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
        # 1️⃣  MONTHLY COMPOSITION
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
        # 2️⃣  YEARLY COMPOSITION
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
        # 3️⃣  SUMMARY TABLE
        # ============================================================
        st.markdown("### 🧾 Summary of Optimal K Results") 

        summary = []

        if best_k_monthly is not None:
            summary.append([
                "Monthly (Freshwater + Marine)",
                best_k_monthly,
                f"{best_sil_monthly:.3f}" if best_sil_monthly else "N/A"
            ])

        if best_k_yearly is not None:
            summary.append([
                "Yearly (Freshwater + Marine)",
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

        # ======================================
        # PAGE HEADER
        # ======================================
        st.markdown("""
            <h2 style='color:white;'>🎣 Fish Landing Trends (Cluster Analysis)</h2>
            <p style='color:#ccc; margin-top:-10px;'>
                Compare freshwater & marine fish landings across yearly or monthly periods using K-Means cluster grouping.
            </p>
        """, unsafe_allow_html=True)

        st.markdown("<hr style='border:0.5px solid #444;'>", unsafe_allow_html=True)

        # ======================================
        # OPTIONS SECTION
        # ======================================
        with st.container():

            st.markdown(
                "<p style='color:#ccc; margin-top:-12px; font-size:15px;'>"
                "Please select the period and trend to display the fish landing analysis."
                "</p>",
                unsafe_allow_html=True
            )

            opt_col1, opt_col2 = st.columns([1, 2])

            with opt_col1:
                period_choice = st.radio("Period:", ["Yearly", "Monthly"], horizontal=True)

            with opt_col2:
                trend_option = st.radio(
                    "Trend:", ["Freshwater", "Marine", "Both"], horizontal=True
                )

        # ======================================
        # YEARLY VIEW
        # ======================================
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

            fw_latest = safe_get(yearly, latest_year, "Freshwater (Tonnes)")
            fw_prev = safe_get(yearly, prev_year, "Freshwater (Tonnes)")
            ma_latest = safe_get(yearly, latest_year, "Marine (Tonnes)")
            ma_prev = safe_get(yearly, prev_year, "Marine (Tonnes)")

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

            # ======================================
            # YEARLY CLUSTER PLOT
            # ======================================
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

        # ======================================
        # MONTHLY VIEW
        # ======================================
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

            # ======================================
            # MONTHLY CLUSTER PLOT
            # ======================================
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



            #  EXPANDABLE CLUSTER INTERPRETATION SECTION
            
            if "yearly_profiles" not in locals():
                # Meaning: the yearly cluster profiles did NOT get created
                # (e.g., monthly view is selected)
                pass
            else:
                with st.expander("📘 Understanding the Clusters (Click to Expand)", expanded=False):

                    st.markdown("""
                        <p style="color:#ccc; font-size:15px;">
                        Each cluster groups years with similar freshwater & marine landing characteristics.
                        The interpretation below helps you understand their economic significance.
                        </p>
                    """, unsafe_allow_html=True)

                    for _, row in yearly_profiles.iterrows():
                        cl = int(row["Cluster"])
                        fw = row["Freshwater (Tonnes)"]
                        ma = row["Marine (Tonnes)"]
                        meaning = row["Meaning"]
                        title = row["Friendly"]

                        cluster_color = [
                            "#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b2"
                        ][cl % 5]

                        st.markdown(f"""
                        <div style="
                            background-color:#111111;
                            border-left:6px solid {cluster_color};
                            padding:16px;
                            border-radius:10px;
                            margin-bottom:14px;
                            color:white;
                        ">

                            <div style="font-size:20px; font-weight:700; color:{cluster_color}">
                                Cluster {cl}: {title}
                            </div>

                            <div style="margin-top:6px; font-size:15px;">
                                <b>Meaning:</b> {meaning}<br><br>

                                <span style="color:#ccc;">Average Freshwater Landing:</span>
                                <b>{fw:,.0f}</b> tonnes<br>

                                <span style="color:#ccc;">Average Marine Landing:</span>
                                <b>{ma:,.0f}</b> tonnes<br>
                            </div>

                            <div style="margin-top:10px; font-size:14px; color:#aaa;">
                                🛈 This cluster appears above with the same color + line style.<br>
                                Understanding cluster patterns helps identify strong & weak production periods.
                            </div>

                        </div>
                        """, unsafe_allow_html=True)



 def calc_growth_month_html(curr, prev):
                if prev == 0:
                    return "<span style='color:gray'>–</span>"
                ratio = curr / prev
                if ratio >= 1:
                    return f"<span style='color:lightgreen'>↑ {ratio:.2f}x</span>"
                else:
                    return f"<span style='color:#ff4d4d'>↓ {ratio:.2f}x</span>"
                



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

        # ======================================
        # PAGE HEADER
        # ======================================
        st.markdown("""
            <h2 style='color:white;'>🎣 Fish Landing Trends (Cluster Analysis)</h2>
            <p style='color:#ccc; margin-top:-10px;'>
                Compare freshwater & marine fish landings across yearly or monthly periods using K-Means cluster grouping.
            </p>
        """, unsafe_allow_html=True)

        st.markdown("<hr style='border:0.5px solid #444;'>", unsafe_allow_html=True)

        # ======================================
        # OPTIONS SECTION
        # ======================================
        with st.container():

            st.markdown(
                "<p style='color:#ccc; margin-top:-12px; font-size:15px;'>"
                "Please select the period and trend to display the fish landing analysis."
                "</p>",
                unsafe_allow_html=True
            )

            opt_col1, opt_col2 = st.columns([1, 2])

            with opt_col1:
                period_choice = st.radio("Period:", ["Yearly", "Monthly"], horizontal=True)

            with opt_col2:
                trend_option = st.radio(
                    "Trend:", ["Freshwater", "Marine", "Both"], horizontal=True
                )


        # ============================================
        # HELPER FUNCTIONS FOR CLUSTER MEANINGS
        # ============================================
 
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


        # ======================================
        # YEARLY VIEW
        # ======================================
        
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

            fw_latest = safe_get(yearly, latest_year, "Freshwater (Tonnes)")
            fw_prev = safe_get(yearly, prev_year, "Freshwater (Tonnes)")
            ma_latest = safe_get(yearly, latest_year, "Marine (Tonnes)")
            ma_prev = safe_get(yearly, prev_year, "Marine (Tonnes)")

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
      

        if period_choice == "Yearly":

            # ============================
            # PREPARE YEARLY DATA
            # ============================
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
                "Marine": "Marine (Tonnes)",
            }, inplace=True)

            # Cluster
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

            # ============================
            # CASE A: BOTH → DUAL AXIS
            # ============================
            if trend_option == "Both":
                fig, ax1 = plt.subplots(figsize=(14, 6))
                ax2 = ax1.twinx()

                # Freshwater (left axis)
                for cl in sorted(melted["Cluster"].unique()):
                    sub = melted[(melted["Type"] == "Freshwater (Tonnes)") & (melted["Cluster"] == cl)]
                    if len(sub):
                        ax1.plot(
                            sub["Year"], sub["Landing"],
                            linestyle=linestyles[cl % len(linestyles)],
                            marker="o", color="tab:blue", markersize=6,
                            label=f"Freshwater – Cluster {cl}"
                        )

                # Marine (right axis)
                for cl in sorted(melted["Cluster"].unique()):
                    sub = melted[(melted["Type"] == "Marine (Tonnes)") & (melted["Cluster"] == cl)]
                    if len(sub):
                        ax2.plot(
                            sub["Year"], sub["Landing"],
                            linestyle=linestyles[cl % len(linestyles)],
                            marker="^", color="tab:red", markersize=6,
                            label=f"Marine – Cluster {cl}"
                        )

                ax1.set_ylabel("Freshwater Landing (Tonnes)", color="tab:blue")
                ax2.set_ylabel("Marine Landing (Tonnes)", color="tab:red")

                ax1.tick_params(axis="y", labelcolor="tab:blue")
                ax2.tick_params(axis="y", labelcolor="tab:red")

                ax1.set_title(f"Yearly Fish Landing Trends (k={best_k})")
                ax1.grid(True, alpha=0.3)

                # Combined legend
                h1, l1 = ax1.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax1.legend(h1 + h2, l1 + l2, loc="upper center",
                        bbox_to_anchor=(0.5, -0.15), ncol=4)

                st.pyplot(fig)

            # ============================
            # CASE B: FRESHWATER ONLY
            # ============================
            elif trend_option == "Freshwater":
                fig, ax = plt.subplots(figsize=(14, 6))

                for cl in sorted(melted["Cluster"].unique()):
                    sub = melted[(melted["Type"] == "Freshwater (Tonnes)")
                                & (melted["Cluster"] == cl)]
                    if len(sub):
                        ax.plot(
                            sub["Year"], sub["Landing"],
                            linestyle=linestyles[cl % len(linestyles)],
                            marker="o", color="tab:blue", markersize=6,
                            label=f"Freshwater – Cluster {cl}"
                        )

                ax.set_ylabel("Freshwater Landing (Tonnes)")
                ax.set_title(f"Yearly Fish Landing Trends (Freshwater Only, k={best_k})")
                ax.grid(True, alpha=0.3)
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4)

                st.pyplot(fig)

            # ============================
            # CASE C: MARINE ONLY
            # ============================
            else:
                fig, ax = plt.subplots(figsize=(14, 6))

                for cl in sorted(melted["Cluster"].unique()):
                    sub = melted[(melted["Type"] == "Marine (Tonnes)")
                                & (melted["Cluster"] == cl)]
                    if len(sub):
                        ax.plot(
                            sub["Year"], sub["Landing"],
                            linestyle=linestyles[cl % len(linestyles)],
                            marker="^", color="tab:red", markersize=6,
                            label=f"Marine – Cluster {cl}"
                        )

                ax.set_ylabel("Marine Landing (Tonnes)")
                ax.set_title(f"Yearly Fish Landing Trends (Marine Only, k={best_k})")
                ax.grid(True, alpha=0.3)
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4)

                st.pyplot(fig)

        else:
             # ======================================
            # MONTHLY VIEW
            # ======================================

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

            latest_date = monthly["MonthYear"].max()
            prev_date = latest_date - pd.DateOffset(months=1)

            def safe_month_value(df, date, col):
                v = df.loc[df["MonthYear"] == date, col]
                return v.values[0] if len(v) else 0

            def calc_growth_month_html(curr, prev):
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


            fw = safe_month_value(monthly, latest_date, "Freshwater (Tonnes)")
            fw_prev = safe_month_value(monthly, prev_date, "Freshwater (Tonnes)")
            ma = safe_month_value(monthly, latest_date, "Marine (Tonnes)")
            ma_prev = safe_month_value(monthly, prev_date, "Marine (Tonnes)")

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

            # Create proper datetime column for plotting
            monthly["MonthYear"] = pd.to_datetime(
                monthly["Year"].astype(str) + "-" + monthly["Month"].astype(str) + "-01"
            )

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

            # ======================================================
            # CASE 1 — BOTH (Dual Y-Axis)
            # ======================================================
            if trend_option == "Both":

                fig, ax1 = plt.subplots(figsize=(14, 6))
                ax2 = ax1.twinx()

                # ---- Freshwater (blue, left) ----
                for cl in sorted(melted["Cluster"].unique()):
                    fw_subset = melted[
                        (melted["Type"] == "Freshwater (Tonnes)") &
                        (melted["Cluster"] == cl)
                    ]

                    if len(fw_subset):
                        ax1.plot(
                            fw_subset["MonthYear"],
                            fw_subset["Landing"],
                            linestyle=linestyles[cl % len(linestyles)],
                            marker="o",
                            color="tab:blue",
                            markersize=5,
                            label=f"Freshwater – Cluster {cl}",
                        )

                # ---- Marine (red, right) ----
                for cl in sorted(melted["Cluster"].unique()):
                    ma_subset = melted[
                        (melted["Type"] == "Marine (Tonnes)") &
                        (melted["Cluster"] == cl)
                    ]

                    if len(ma_subset):
                        ax2.plot(
                            ma_subset["MonthYear"],
                            ma_subset["Landing"],
                            linestyle=linestyles[cl % len(linestyles)],
                            marker="^",
                            color="tab:red",
                            markersize=5,
                            label=f"Marine – Cluster {cl}",
                        )

                # Axis labels
                ax1.set_ylabel("Freshwater Landing (Tonnes)", color="tab:blue")
                ax2.set_ylabel("Marine Landing (Tonnes)", color="tab:red")

                ax1.tick_params(axis="y", labelcolor="tab:blue")
                ax2.tick_params(axis="y", labelcolor="tab:red")

                ax1.set_title(f"Monthly Fish Landing Trends (k={best_k})")
                ax1.grid(True, alpha=0.3)

                # Combine legends
                h1, l1 = ax1.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()

                ax1.legend(
                    h1 + h2,
                    l1 + l2,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.18),
                    ncol=4
                )

                plt.xticks(rotation=45)

                st.pyplot(fig)

            # ======================================================
            # CASE 2 — FRESHWATER ONLY
            # ======================================================
            elif trend_option == "Freshwater":

                fig, ax = plt.subplots(figsize=(14, 6))

                for cl in sorted(melted["Cluster"].unique()):
                    sub = melted[
                        (melted["Type"] == "Freshwater (Tonnes)") &
                        (melted["Cluster"] == cl)
                    ]

                    if len(sub):
                        ax.plot(
                            sub["MonthYear"],
                            sub["Landing"],
                            linestyle=linestyles[cl % len(linestyles)],
                            marker="o",
                            color="tab:blue",
                            markersize=5,
                            label=f"Freshwater – Cluster {cl}",
                        )

                ax.set_ylabel("Freshwater Landing (Tonnes)", color="tab:blue")
                ax.set_title(f"Monthly Fish Landing Trends (Freshwater Only, k={best_k})")
                ax.grid(True, alpha=0.3)
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4)

                plt.xticks(rotation=45)

                st.pyplot(fig)

            # ======================================================
            # CASE 3 — MARINE ONLY
            # ======================================================
            else:  # trend_option == "Marine"

                fig, ax = plt.subplots(figsize=(14, 6))

                for cl in sorted(melted["Cluster"].unique()):
                    sub = melted[
                        (melted["Type"] == "Marine (Tonnes)") &
                        (melted["Cluster"] == cl)
                    ]

                    if len(sub):
                        ax.plot(
                            sub["MonthYear"],
                            sub["Landing"],
                            linestyle=linestyles[cl % len(linestyles)],
                            marker="^",
                            color="tab:red",
                            markersize=5,
                            label=f"Marine – Cluster {cl}",
                        )

                ax.set_ylabel("Marine Landing (Tonnes)", color="tab:red")
                ax.set_title(f"Monthly Fish Landing Trends (Marine Only, k={best_k})")
                ax.grid(True, alpha=0.3)
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4)

                plt.xticks(rotation=45)

                st.pyplot(fig)




                if period_choice == "Yearly":

            # ============================
            # PREPARE YEARLY DATA
            # ============================
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
                "Marine": "Marine (Tonnes)",
            }, inplace=True)

            # Cluster
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

            # ============================
            # CASE A: BOTH → DUAL AXIS
            # ============================
            if trend_option == "Both":
                fig, ax1 = plt.subplots(figsize=(14, 6))
                ax2 = ax1.twinx()

                # Freshwater (left axis)
                for cl in sorted(melted["Cluster"].unique()):
                    sub = melted[(melted["Type"] == "Freshwater (Tonnes)") & (melted["Cluster"] == cl)]
                    if len(sub):
                        ax1.plot(
                            sub["Year"], sub["Landing"],
                            linestyle=linestyles[cl % len(linestyles)],
                            marker="o", color="tab:blue", markersize=6,
                            label=f"Freshwater – Cluster {cl}"
                        )

                # Marine (right axis)
                for cl in sorted(melted["Cluster"].unique()):
                    sub = melted[(melted["Type"] == "Marine (Tonnes)") & (melted["Cluster"] == cl)]
                    if len(sub):
                        ax2.plot(
                            sub["Year"], sub["Landing"],
                            linestyle=linestyles[cl % len(linestyles)],
                            marker="^", color="tab:red", markersize=6,
                            label=f"Marine – Cluster {cl}"
                        )

                # =============================
                # HIGHLIGHT SELECTED YEAR POINT
                # =============================
                try:
                    # Highlight freshwater use ax1
                    fw_y = yearly.loc[yearly["Year"] == selected_year, "Freshwater (Tonnes)"].values[0]
                    ax1.scatter(
                        selected_year, fw_y,
                        color="yellow", edgecolor="black",
                        s=180, zorder=10, label="Selected Year"
                    )

                    # Highlight marine use ax2
                    ma_y = yearly.loc[yearly["Year"] == selected_year, "Marine (Tonnes)"].values[0]
                    ax2.scatter(
                        selected_year, ma_y,
                        color="yellow", edgecolor="black",
                        s=180, zorder=10
                    )
                except:
                    pass

                ax1.set_ylabel("Freshwater Landing (Tonnes)", color="tab:blue")
                ax2.set_ylabel("Marine Landing (Tonnes)", color="tab:red")

                ax1.tick_params(axis="y", labelcolor="tab:blue")
                ax2.tick_params(axis="y", labelcolor="tab:red")

                ax1.set_title(f"Yearly Fish Landing Trends (k={best_k})")
                ax1.grid(True, alpha=0.3)

                #Hover tooltip

                annot = ax1.annotate("", xy=(0,0), xytext=(20,20),
                     textcoords="offset points", fontsize=10,
                     bbox=dict(boxstyle="round", fc="yellow", ec="black"),
                     arrowprops=dict(arrowstyle="->"))
                annot.set_visible(False)

                def hover(event):
                    if event.inaxes == ax1:
                        for cl in sorted(melted["Cluster"].unique()):
                            sub = melted[(melted["Type"] == "Freshwater (Tonnes)") & (melted["Cluster"] == cl)]
                            for x, y in zip(sub["Year"], sub["Landing"]):
                                if abs(x - event.xdata) < 0.3 and abs(y - event.ydata) < (y * 0.05):
                                    annot.xy = (x, y)
                                    annot.set_text(f"Year: {int(x)}\nLanding: {y:,.0f} tonnes")
                                    annot.set_visible(True)
                                    fig.canvas.draw_idle()
                                    return
                    annot.set_visible(False)

                fig.canvas.mpl_connect("motion_notify_event", hover)


                # Combined legend
                h1, l1 = ax1.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax1.legend(h1 + h2, l1 + l2, loc="upper center",
                        bbox_to_anchor=(0.5, -0.15), ncol=4)

                st.pyplot(fig)

            # ============================
            # CASE B: FRESHWATER ONLY
            # ============================
            elif trend_option == "Freshwater":
                fig, ax = plt.subplots(figsize=(14, 6))

                for cl in sorted(melted["Cluster"].unique()):
                    sub = melted[(melted["Type"] == "Freshwater (Tonnes)")
                                & (melted["Cluster"] == cl)]
                    if len(sub):
                        ax.plot(
                            sub["Year"], sub["Landing"],
                            linestyle=linestyles[cl % len(linestyles)],
                            marker="o", color="tab:blue", markersize=6,
                            label=f"Freshwater – Cluster {cl}"
                        )


                # Highlight selected year
                try:
                    fw_y = yearly.loc[yearly["Year"] == selected_year, "Freshwater (Tonnes)"].values[0]
                    ax.scatter(
                        selected_year, fw_y,
                        color="yellow", edgecolor="black",
                        s=180, zorder=10, label="Selected Year"
                    )
                except:
                    pass

                ax.set_ylabel("Freshwater Landing (Tonnes)")
                ax.set_title(f"Yearly Fish Landing Trends (Freshwater Only, k={best_k})")

                annot = ax.annotate("", xy=(0,0), xytext=(20,20),
                     textcoords="offset points", fontsize=10,
                     bbox=dict(boxstyle="round", fc="yellow", ec="black"),
                     arrowprops=dict(arrowstyle="->"))
                annot.set_visible(False)

                def hover(event):
                    if event.inaxes == ax:
                        for cl in sorted(melted["Cluster"].unique()):
                            sub = melted[(melted["Type"] == "Freshwater (Tonnes)") & (melted["Cluster"] == cl)]
                            for x, y in zip(sub["Year"], sub["Landing"]):
                                if abs(x - event.xdata) < 0.3 and abs(y - event.ydata) < (y * 0.05):
                                    annot.xy = (x, y)
                                    annot.set_text(f"Year: {int(x)}\nLanding: {y:,.0f} tonnes")
                                    annot.set_visible(True)
                                    fig.canvas.draw_idle()
                                    return
                    annot.set_visible(False)

                fig.canvas.mpl_connect("motion_notify_event", hover)

                ax.grid(True, alpha=0.3)
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4)

                st.pyplot(fig)

            # ============================
            # CASE C: MARINE ONLY
            # ============================
            else:
                fig, ax = plt.subplots(figsize=(14, 6))

                for cl in sorted(melted["Cluster"].unique()):
                    sub = melted[(melted["Type"] == "Marine (Tonnes)")
                                & (melted["Cluster"] == cl)]
                    if len(sub):
                        ax.plot(
                            sub["Year"], sub["Landing"],
                            linestyle=linestyles[cl % len(linestyles)],
                            marker="^", color="tab:red", markersize=6,
                            label=f"Marine – Cluster {cl}"
                        )

                
                try:
                    ma_y = yearly.loc[yearly["Year"] == selected_year, "Marine (Tonnes)"].values[0]
                    ax.scatter(
                        selected_year, ma_y,
                        color="yellow", edgecolor="black",
                        s=180, zorder=10, label="Selected Year"
                    )
                except:
                    pass

                ax.set_ylabel("Marine Landing (Tonnes)")
                ax.set_title(f"Yearly Fish Landing Trends (Marine Only, k={best_k})")

                annot = ax.annotate("", xy=(0,0), xytext=(20,20),
                     textcoords="offset points", fontsize=10,
                     bbox=dict(boxstyle="round", fc="yellow", ec="black"),
                     arrowprops=dict(arrowstyle="->"))
                annot.set_visible(False)

                def hover(event):
                    if event.inaxes == ax:
                        for cl in sorted(melted["Cluster"].unique()):
                            sub = melted[(melted["Type"] == "Freshwater (Tonnes)") & (melted["Cluster"] == cl)]
                            for x, y in zip(sub["Year"], sub["Landing"]):
                                if abs(x - event.xdata) < 0.3 and abs(y - event.ydata) < (y * 0.05):
                                    annot.xy = (x, y)
                                    annot.set_text(f"Year: {int(x)}\nLanding: {y:,.0f} tonnes")
                                    annot.set_visible(True)
                                    fig.canvas.draw_idle()
                                    return
                    annot.set_visible(False)

                fig.canvas.mpl_connect("motion_notify_event", hover)

                ax.grid(True, alpha=0.3)
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4)

                st.pyplot(fig)




                else:
             # ======================================
            # MONTHLY VIEW
            # ======================================

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

            # Create proper datetime column for indexing
            monthly["MonthYear"] = pd.to_datetime(
                monthly["Year"].astype(str) + "-" + monthly["Month"].astype(str) + "-01"
            )

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

            # ======================================================
            # CASE 1 — BOTH (Dual Y-Axis)
            # ======================================================
            # ============================================================
            # MONTHLY TREND PLOTS (RESPOND TO SELECTED YEAR & MONTH)
            # ============================================================

            # Filter dataset for the selected year only
            monthly_year = monthly[monthly["Year"] == selected_year].copy()

            # KMeans clustering
            features = ["Freshwater (Tonnes)", "Marine (Tonnes)"]
            scaled = StandardScaler().fit_transform(monthly_year[features])

            best_k = st.session_state.get("best_k_monthly", 3)
            monthly_year["Cluster"] = KMeans(n_clusters=best_k, random_state=42).fit_predict(scaled)

            st.markdown(f"**Optimal clusters used:** {best_k}")

            # Melt for plotting
            melted = monthly_year.melt(
                id_vars=["MonthYear", "Cluster"],
                value_vars=["Freshwater (Tonnes)", "Marine (Tonnes)"],
                var_name="Type",
                value_name="Landing",
            )

            # Linestyles
            linestyles = ["solid", "dashed", "dotted", "dashdot"]

            # Highlight selected month
            highlight_date = selected_date


            # ============================================================
            # CASE 1 — BOTH (Dual Axis)
            # ============================================================
            if trend_option == "Both":

                fig, ax1 = plt.subplots(figsize=(14, 6))
                ax2 = ax1.twinx()

                # --- Freshwater (blue, left) ---
                for cl in sorted(melted["Cluster"].unique()):
                    fw_subset = melted[
                        (melted["Type"] == "Freshwater (Tonnes)") &
                        (melted["Cluster"] == cl)
                    ]

                    if len(fw_subset):
                        ax1.plot(
                            fw_subset["MonthYear"],
                            fw_subset["Landing"],
                            linestyle=linestyles[cl % len(linestyles)],
                            marker="o",
                            color="tab:blue",
                            markersize=5,
                            label=f"Freshwater – Cluster {cl}",
                        )

                # --- Marine (red, right) ---
                for cl in sorted(melted["Cluster"].unique()):
                    ma_subset = melted[
                        (melted["Type"] == "Marine (Tonnes)") &
                        (melted["Cluster"] == cl)
                    ]

                    if len(ma_subset):
                        ax2.plot(
                            ma_subset["MonthYear"],
                            ma_subset["Landing"],
                            linestyle=linestyles[cl % len(linestyles)],
                            marker="^",
                            color="tab:red",
                            markersize=5,
                            label=f"Marine – Cluster {cl}",
                        )

                # Highlight selected month
                fw_val = monthly_year.loc[monthly_year["MonthYear"] == highlight_date, "Freshwater (Tonnes)"]
                ma_val = monthly_year.loc[monthly_year["MonthYear"] == highlight_date, "Marine (Tonnes)"]

                if len(fw_val):
                    ax1.scatter(highlight_date, fw_val.values[0], s=180, color="yellow", edgecolor="black", zorder=5)

                if len(ma_val):
                    ax2.scatter(highlight_date, ma_val.values[0], s=180, color="yellow", edgecolor="black", zorder=5)

                ax1.set_ylabel("Freshwater Landing (Tonnes)", color="tab:blue")
                ax2.set_ylabel("Marine Landing (Tonnes)", color="tab:red")

                ax1.tick_params(axis="y", labelcolor="tab:blue")
                ax2.tick_params(axis="y", labelcolor="tab:red")

                ax1.set_title(f"Monthly Fish Landing Trends in {selected_year} (k={best_k})")
                ax1.grid(True, alpha=0.3)

                # Combine legends
                h1, l1 = ax1.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()

                ax1.legend(
                    h1 + h2,
                    l1 + l2,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.18),
                    ncol=4
                )

                plt.xticks(rotation=45)
                st.pyplot(fig)



            # ============================================================
            # CASE 2 — FRESHWATER ONLY
            # ============================================================
            elif trend_option == "Freshwater":

                fig, ax = plt.subplots(figsize=(14, 6))

                for cl in sorted(melted["Cluster"].unique()):
                    sub = melted[
                        (melted["Type"] == "Freshwater (Tonnes)") &
                        (melted["Cluster"] == cl)
                    ]

                    if len(sub):
                        ax.plot(
                            sub["MonthYear"],
                            sub["Landing"],
                            linestyle=linestyles[cl % len(linestyles)],
                            marker="o",
                            color="tab:blue",
                            markersize=5,
                            label=f"Freshwater – Cluster {cl}",
                        )

                # Highlight point
                fw_val = monthly_year.loc[monthly_year["MonthYear"] == highlight_date, "Freshwater (Tonnes)"]
                if len(fw_val):
                    ax.scatter(highlight_date, fw_val.values[0], s=200, color="yellow", edgecolor="black", zorder=5)

                ax.set_ylabel("Freshwater Landing (Tonnes)", color="tab:blue")
                ax.set_title(f"Monthly Fish Landing Trends (Freshwater Only) in {selected_year}")
                ax.grid(True, alpha=0.3)
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4)

                plt.xticks(rotation=45)
                st.pyplot(fig)



            # ============================================================
            # CASE 3 — MARINE ONLY
            # ============================================================
            else:

                fig, ax = plt.subplots(figsize=(14, 6))

                for cl in sorted(melted["Cluster"].unique()):
                    sub = melted[
                        (melted["Type"] == "Marine (Tonnes)") &
                        (melted["Cluster"] == cl)
                    ]

                    if len(sub):
                        ax.plot(
                            sub["MonthYear"],
                            sub["Landing"],
                            linestyle=linestyles[cl % len(linestyles)],
                            marker="^",
                            color="tab:red",
                            markersize=5,
                            label=f"Marine – Cluster {cl}",
                        )

                # Highlight selected month
                ma_val = monthly_year.loc[monthly_year["MonthYear"] == highlight_date, "Marine (Tonnes)"]
                if len(ma_val):
                    ax.scatter(highlight_date, ma_val.values[0], s=200, color="yellow", edgecolor="black", zorder=5)

                ax.set_ylabel("Marine Landing (Tonnes)", color="tab:red")
                ax.set_title(f"Monthly Fish Landing Trends (Marine Only) in {selected_year}")
                ax.grid(True, alpha=0.3)
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4)

                plt.xticks(rotation=45)
                st.pyplot(fig)