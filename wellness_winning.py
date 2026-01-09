import streamlit as st
import plotly.express as px
import numpy as np


def show_wellness_winning(gap_df):
    if gap_df is not None and not gap_df.empty:

        # --- DATA FILTERING ---
        gap_df = gap_df[gap_df['Year'] >= 1920].copy()

        # --- LAYOUT ADJUSTMENT ---
        col_title, col_controls = st.columns([3, 2], gap="medium")

        with col_title:
            st.title("📈 Wellness & Winning Over Time")

        with col_controls:
            st.write("")
            st.write("")
            roi_mode = st.radio(
                "Select View:",
                ["Efficiency (Medals per Million)", "Total Impact (Total Medals)"],
                horizontal=True,
                label_visibility="collapsed"
            )

        # -------------------------

        if roi_mode == "Efficiency (Medals per Million)":
            fig = px.scatter(
                gap_df, x="Life_Expectancy", y="Medals_Per_Million",
                animation_frame="Year", animation_group="Country_Name",
                size="Delegation_Size", color="continent",
                hover_name="Country_Name", size_max=50,
                range_x=[35, 90],
                range_y=[-1.5, 7.5],
                title="1. Health vs Talent (Medals Per Million)"
            )

            fig.update_layout(
                height=700,
                yaxis=dict(
                    tickvals=[0, 2, 4, 6],
                    ticktext=["0", "2", "4", "6"],
                    title="Medals Per Million"
                )
            )

        else:
            # --- MANUAL SPACING (THE "RANK" METHOD) ---

            # 1. Define exactly the ticks you want (Updated List)
            custom_ticks = [1, 3, 7, 12, 20, 30, 50, 80, 200]

            # 2. Create corresponding "Positions" (0, 1, 2, 3...)
            tick_positions = list(range(len(custom_ticks)))

            # 3. Map real medal values to these positions
            def map_to_custom_scale(val):
                if val < 1: return 0
                return np.interp(val, custom_ticks, tick_positions)

            gap_df['Y_Pos_Artificial'] = gap_df['Medals'].apply(map_to_custom_scale)

            # Calculate dynamic range
            max_idx = len(custom_ticks) - 1

            fig = px.scatter(
                gap_df, x="Life_Expectancy", y="Y_Pos_Artificial",
                animation_frame="Year", animation_group="Country_Name",
                size="Delegation_Size", color="continent",
                hover_name="Country_Name", hover_data=["Medals"],
                size_max=50, range_x=[35, 90],

                # Dynamic range to fit all ticks comfortably
                range_y=[-0.5, max_idx + 0.5],

                log_y=False,
                title="2. Health vs Total Medals (Custom Spaced)"
            )

            fig.update_layout(
                height=700,
                yaxis=dict(
                    title="Total Medals",
                    tickvals=tick_positions,
                    ticktext=[str(x) for x in custom_ticks],
                    showgrid=True,
                    gridcolor='#E5E5E5'
                )
            )

        # Slow down animation
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 700

        st.plotly_chart(fig, use_container_width=True)