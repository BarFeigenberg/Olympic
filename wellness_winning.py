import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


def show_wellness_winning(gap_df):
    """
    Display interactive 'Wellness & Winning' chart showing medals/score vs life expectancy.
    Supports dynamic scaling (Absolute / Per Million) and animation over Olympic years.

    @param gap_df: Gapminder-style dataframe with population, medals, life_expectancy, delegation_size, continent
    """
    if gap_df is not None and not gap_df.empty:

        # --- Filter dataset for relevant years (1920+) ---
        gap_df = gap_df[gap_df['year'] >= 1920].copy()

        # --- Define color palette for continents ---
        olympic_palette = {
            # Okabe-Ito Palette (Colorblind Friendly)
            "Europe": "#009E73",    # Bluish Green (תכלת כהה/טורקיז - אירופה)
            "Asia": "#E69F00",      # Orange (כתום - אסיה)
            "Africa": "#F0E442",    # Yellow (צהוב - אפריקה)
            "Americas": "#0072B2",  # Blue (כחול כהה - אמריקה)
            "Oceania": "#56B4E9"    # Sky Blue (תכלת בהיר - אוקיאניה)
        }

        # --- Top Row: Title ---
        st.title("Wellness & Winning Over Time")
        st.divider()

        # --- Controls Row (Left Aligned) ---
        # We use columns [1, 1, 5] to keep buttons compact on the left
        col_ctrl1, col_ctrl2, col_spacer = st.columns([1, 1, 5], gap="small")

        # Metric selection
        with col_ctrl1:
            metric_mode = st.radio(
                "Metric:",
                ["Total Medals", "Weighted Score"],
                horizontal=True,  # Keep them horizontal inside the column
                label_visibility="visible",  # Show the label "Metric:"
                key="wellness_metric"
            )

        # View mode selection
        with col_ctrl2:
            view_mode = st.radio(
                "View:",
                ["Per Million", "Absolute"],
                horizontal=True,
                label_visibility="visible",  # Show the label "View:"
                key="wellness_view"
            )

        # --- Define annotation style for overlaying the year in chart ---
        def get_year_annotation(year_value):
            return dict(
                text=str(year_value),
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=200, color="rgba(200, 200, 200, 0.2)")
            )

        # --- Chart Configuration ---
        # Determine which metric column to use
        if metric_mode == "Weighted Score":
            base_col = "score"
            display_name = "Weighted Score"
        else:
            base_col = "medals"
            display_name = "Total Medals"

        # Compute plotting values based on view mode
        if view_mode == "Per Million":
            gap_df['plot_value'] = (gap_df[base_col] / gap_df['population']) * 1_000_000
            chart_title_text = f"Health vs Efficiency ({display_name} Per Million)"
            y_axis_title = f"{display_name} / Million (Non-Linear)"
            tooltip_label = f"{display_name} (Per 1M)"
            # Custom tick configuration for non-linear y-axis
            custom_ticks = [0, 1, 2, 5, 15, 120] if metric_mode == "Weighted Score" else [0, 1, 2, 4, 10, 20, 40, 60]
        else:
            gap_df['plot_value'] = gap_df[base_col]
            chart_title_text = f"Health vs {display_name}"
            y_axis_title = f"{display_name} (Non-Linear)"
            tooltip_label = display_name
            custom_ticks = [2, 6, 14, 24, 40, 60, 100, 160, 500] if metric_mode == "Weighted Score" else [1, 3, 7, 12,
                                                                                                          20, 30, 50,
                                                                                                          80, 200]

        # --- Display chart header with bubble size explanation ---
        st.markdown(
            f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; margin-top: 25px;">
                    <h3 style="margin: 0; font-size: 1.5rem;">{chart_title_text}</h3>
                    <span style="font-size: 16px; color: #555;">Bubble size = Delegation size</span>
                </div>
                """,
            unsafe_allow_html=True
        )

        # --- Non-linear scaling helper ---
        tick_positions = list(range(len(custom_ticks)))

        def map_to_custom_scale(val):
            """Map raw plot_value to custom tick positions for non-linear scaling."""
            if val < custom_ticks[0]: return 0
            if val > custom_ticks[-1]: return len(custom_ticks) - 1
            return np.interp(val, custom_ticks, tick_positions)

        gap_df['y_pos_scaled'] = gap_df['plot_value'].apply(map_to_custom_scale)
        max_idx = len(custom_ticks) - 1

        # --- Create scatter plot ---
        fig = px.scatter(
            gap_df, x="life_expectancy", y="y_pos_scaled",
            animation_frame="year", animation_group="country_name",
            size="delegation_size", color="continent",
            color_discrete_map=olympic_palette,
            hover_name="country_name",
            labels={
                "plot_value": tooltip_label,
                "life_expectancy": "Life Exp. (Years)",
                "delegation_size": "Delegation Size",
                "continent": "Continent",
                "year": "Year"
            },
            hover_data={"y_pos_scaled": False, "plot_value": ':.1f', "delegation_size": True},
            size_max=50,
            range_x=[35, 90],
            range_y=[-0.5, max_idx + 0.5]
        )

        # --- Layout updates ---
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

        # Consistent marker transparency and outline
        fig.update_traces(marker=dict(opacity=0.7, line=dict(width=1, color='White')))

        # --- Overlay year annotation for animation frames ---
        initial_year = gap_df['year'].min()
        fig.update_layout(annotations=[get_year_annotation(initial_year)])

        if fig.frames:
            for frame in fig.frames:
                frame_year = frame.name
                frame.layout = go.Layout(annotations=[get_year_annotation(frame_year)])

        # --- Animation configuration ---
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 700
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["redraw"] = True
        fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 300

        # Ensure sliders redraw on each frame
        if fig.layout.sliders:
            for slider in fig.layout.sliders:
                for step in slider.steps:
                    args_list = list(step.args)
                    args_list[1] = {
                        "frame": {"duration": 700, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 300}
                    }
                    step.args = args_list

        # --- Display Plotly chart in Streamlit ---
        st.plotly_chart(fig, width='stretch')
