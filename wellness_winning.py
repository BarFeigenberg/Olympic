import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


def show_wellness_winning(gap_df):
    if gap_df is not None and not gap_df.empty:

        # --- DATA FILTERING ---
        gap_df = gap_df[gap_df['year'] >= 1920].copy()

        # Olympic Color Map for Continents
        olympic_palette = {
            "Europe": "#0081C8",  # Blue
            "Asia": "#00A651",  # Green
            "Africa": "#000000",  # Black
            "Americas": "#EE334E",  # Red
            "Oceania": "#FCB131"  # Yellow
        }

        # --- CSS to push radio buttons to the right ---
        st.markdown("""
            <style>
                /* Push radio buttons to far right, vertically centered */
                div[data-testid="stHorizontalBlock"]:first-child div[data-testid="column"]:last-child {
                    display: flex;
                    justify-content: flex-end;
                    align-items: center;
                }
            </style>
        """, unsafe_allow_html=True)

        # --- TOP ROW: Page Title + Controls ---
        # We need two controls: Metric (Score/Total) and View (Absolute/Per Million)
        col_title, col_ctrl1, col_ctrl2 = st.columns([2, 1, 1], gap="medium")

        with col_title:
            st.title("Wellness & Winning Over Time")

        with col_ctrl1:
            metric_mode = st.radio(
                "Metric:",
                ["Total Medals", "Weighted Score"],
                horizontal=True,
                label_visibility="collapsed",
                key="wellness_metric"
            )

        with col_ctrl2:
            view_mode = st.radio(
                "View:",
                ["Absolute", "Per Million"],
                horizontal=True,
                label_visibility="collapsed",
                key="wellness_view"
            )

        # -------------------------
        # DEFINING ANNOTATION STYLES
        # -------------------------
        def get_year_annotation(year_value):
            return dict(
                text=str(year_value),
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=200, color="rgba(200, 200, 200, 0.2)")
            )

        # --- CHART LOGIC CONFIGURATION ---
        
        # 1. Determine Value Column
        if metric_mode == "Weighted Score":
             base_col = "score"
             display_name = "Weighted Score"
        else:
             base_col = "medals"
             display_name = "Total Medals"

        if view_mode == "Per Million":
            # Must calculate on the fly as gap_df usually only has 'medals_per_million' pre-calculated
            # We calculate specific per-million based on selection
             gap_df['plot_value'] = (gap_df[base_col] / gap_df['population']) * 1_000_000
             chart_title_text = f"Health vs Efficiency ({display_name} Per Million)"
             y_axis_title = f"{display_name} / Million (Non-Linear)"
             
             # Scale Configuration (Per Million)
             if metric_mode == "Weighted Score":
                  # Score/Million - User requested specific scale (5, 15, remove 25)
                  custom_ticks = [0, 1, 2, 5, 15, 120]
             else:
                  # Total/Million
                  custom_ticks = [0, 1, 2, 4, 10, 20, 40, 60]
                  
        else:
            # Absolute
             gap_df['plot_value'] = gap_df[base_col]
             chart_title_text = f"Health vs {display_name}"
             y_axis_title = f"{display_name} (Non-Linear)"
             
             # Scale Configuration (Absolute)
             if metric_mode == "Weighted Score":
                  # Score Absolute - 2x the Total scale (roughly), max 500
                  custom_ticks = [2, 6, 14, 24, 40, 60, 100, 160, 500]
             else:
                  # Total Absolute
                  custom_ticks = [1, 3, 7, 12, 20, 30, 50, 80, 200]


        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <h3 style="margin: 0; font-size: 1.5rem;">{chart_title_text}</h3>
                <span style="font-size: 16px; color: #555;">Bubble size = Delegation size</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        # --- NON-LINEAR SCALING ---
        tick_positions = list(range(len(custom_ticks)))

        def map_to_custom_scale(val):
            # Safe mapping for non-linear scale logic
            if val < custom_ticks[0]: return 0
            if val > custom_ticks[-1]: return len(custom_ticks) - 1 # Cap at max
            return np.interp(val, custom_ticks, tick_positions)

        gap_df['y_pos_scaled'] = gap_df['plot_value'].apply(map_to_custom_scale)
        max_idx = len(custom_ticks) - 1

        # --- PLOTTING ---
        fig = px.scatter(
            gap_df, x="life_expectancy", y="y_pos_scaled",
            animation_frame="year", animation_group="country_name",
            size="delegation_size", color="continent",
            color_discrete_map=olympic_palette,
            hover_name="country_name",
            hover_data={"y_pos_scaled": False, "plot_value": ':.1f', "delegation_size": True},
            size_max=50,
            range_x=[35, 90],
            range_y=[-0.5, max_idx + 0.5]
        )

        fig.update_layout(
            height=700,
            yaxis=dict(
                tickvals=tick_positions,
                ticktext=[str(x) for x in custom_ticks],
                title=y_axis_title,
                showgrid=True,
                gridcolor='#E5E5E5'
            )
        )
        # Ensure consistent transparency for all points
        fig.update_traces(marker=dict(opacity=0.7, line=dict(width=1, color='White')))

        # -------------------------
        # ANIMATION FRAME UPDATES
        # -------------------------
        initial_year = gap_df['year'].min()
        fig.update_layout(
            annotations=[
                get_year_annotation(initial_year)
            ]
        )

        if fig.frames:
            for frame in fig.frames:
                frame_year = frame.name
                frame.layout = go.Layout(
                    annotations=[
                        get_year_annotation(frame_year)
                    ]
                )

        # Configuration for smoother animation
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 700
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["redraw"] = True
        fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 300
        
        # Also update sliders to redraw on each step
        if fig.layout.sliders:
            for slider in fig.layout.sliders:
                for step in slider.steps:
                    args_list = list(step.args)
                    args_list[1] = {"frame": {"duration": 700, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 300}}
                    step.args = args_list

        st.plotly_chart(fig, width='stretch')
