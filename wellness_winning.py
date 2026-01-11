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

        # --- LAYOUT ADJUSTMENT ---
        col_title, col_controls = st.columns([3, 2], gap="medium")

        with col_title:
            st.title("Wellness & Winning Over Time")

        with col_controls:
            st.write("")
            st.write("")
            roi_mode = st.radio(
                "Select View:",
                ["Medals per Million)", "Total Medals"],
                horizontal=True,
                label_visibility="collapsed"
            )

        # -------------------------
        # DEFINING ANNOTATION STYLES
        # -------------------------

        # Function to create the Background Year Annotation
        def get_year_annotation(year_value):
            return dict(
                text=str(year_value),
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=200, color="rgba(200, 200, 200, 0.2)")  # Large transparent gray
            )

        # Helper string for the subtitle next to the title
        bubble_legend_text = ("<span style='font-size: 14px; color: #555; font-weight: normal;'>                                                            "
                              "Bubble size = Delegation size</span>")

        if roi_mode == "Efficiency (Medals per Million)":
            # --- EFFICIENCY VIEW ---
            # Updated ticks as requested: removed 3, 6, 10
            custom_ticks_eff = [0, 1, 2, 4, 20]
            tick_positions_eff = list(range(len(custom_ticks_eff)))

            def map_efficiency(val):
                if val < 0: return 0
                return np.interp(val, custom_ticks_eff, tick_positions_eff)

            gap_df['y_pos_efficiency'] = gap_df['medals_per_million'].apply(map_efficiency)
            max_idx_eff = len(custom_ticks_eff) - 1

            fig = px.scatter(
                gap_df, x="life_expectancy", y="y_pos_efficiency",
                animation_frame="year", animation_group="country_name",
                size="delegation_size", color="continent",
                color_discrete_map=olympic_palette,
                hover_name="country_name",
                hover_data={"y_pos_efficiency": False, "medals_per_million": ':.2f'},
                size_max=50,
                range_x=[35, 90],
                range_y=[-0.5, max_idx_eff + 0.5],
                title=f"Health vs Talent (Medals Per Million - Non-Linear Scale){bubble_legend_text}"
            )

            fig.update_layout(
                height=700,
                yaxis=dict(
                    tickvals=tick_positions_eff,
                    ticktext=[str(x) for x in custom_ticks_eff],
                    title="Medals Per Million (Non-Linear Scale)",
                    showgrid=True,
                    gridcolor='#E5E5E5'
                )
            )

        else:
            # --- TOTAL MEDALS VIEW ---
            custom_ticks = [1, 3, 7, 12, 20, 30, 50, 80, 200]
            tick_positions = list(range(len(custom_ticks)))

            def map_to_custom_scale(val):
                if val < 1: return 0
                return np.interp(val, custom_ticks, tick_positions)

            gap_df['y_pos_artificial'] = gap_df['medals'].apply(map_to_custom_scale)
            max_idx = len(custom_ticks) - 1

            fig = px.scatter(
                gap_df, x="life_expectancy", y="y_pos_artificial",
                animation_frame="year", animation_group="country_name",
                size="delegation_size", color="continent",
                color_discrete_map=olympic_palette,
                hover_name="country_name", hover_data=["medals"],
                size_max=50, range_x=[35, 90],
                range_y=[-0.5, max_idx + 0.5],
                log_y=False,
                title=f"Health vs Total Medals (Non-Linear Scale){bubble_legend_text}"
            )

            fig.update_layout(
                height=700,
                yaxis=dict(
                    title="Total Medals (Non-Linear Scale)",
                    tickvals=tick_positions,
                    ticktext=[str(x) for x in custom_ticks],
                    showgrid=True,
                    gridcolor='#E5E5E5'
                )
            )
            # Ensure consistent transparency for all points
            fig.update_traces(marker=dict(opacity=0.7, line=dict(width=1, color='White')))

        # -------------------------
        # ANIMATION FRAME UPDATES
        # -------------------------

        # 1. Set the Initial Layout (Start Year only)
        initial_year = gap_df['year'].min()
        # New Legend Annotation
        bubble_legend = dict(
            x=1, y=-0.12,  # Positioned at the bottom right, below the axis
            xref="paper", yref="paper",
            text="⚪ <b>Bubble size</b> = Delegation size",
            showarrow=False,
            font=dict(size=12, color="#555"),
            xanchor="right"
        )

        fig.update_layout(
            annotations=[
                get_year_annotation(initial_year)
            ]
        )

        # 2. Force Update on Every Animation Frame
        if fig.frames:
            for frame in fig.frames:
                frame_year = frame.name

                # Keep BOTH annotations: the changing year and the static legend
                frame.layout = go.Layout(
                    annotations=[
                        get_year_annotation(frame_year),
                        bubble_legend
                    ]
                )

        # Adjust animation speed and force redraw for annotations
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 700
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["redraw"] = True
        fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 300

        # Also update sliders to redraw on each step
        if fig.layout.sliders:
            for slider in fig.layout.sliders:
                for step in slider.steps:
                    # Convert tuple to list to allow modification
                    args_list = list(step.args)
                    args_list[1] = {"frame": {"duration": 700, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 300}}
                    step.args = args_list

        st.plotly_chart(fig, width='stretch')
