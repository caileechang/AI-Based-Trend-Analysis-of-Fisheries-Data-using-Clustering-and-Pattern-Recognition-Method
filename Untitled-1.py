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
