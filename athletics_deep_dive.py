import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

def show_athletics_deep_dive(athletics_df):
    st.title("🏃 Athletics Analysis")

    if athletics_df is not None and not athletics_df.empty:
        # Basic data cleaning
        athletics_df['Gender'] = athletics_df['Gender'].astype(str).str.strip().str.upper()

        events = sorted(athletics_df['BaseEvent'].unique().tolist())
        e_name = st.selectbox("Select Event:", events)
        mode = st.selectbox("Display:", ['Men Only', 'Women Only', 'Both'])

        # Filter data by selected event
        vdf = athletics_df[athletics_df['BaseEvent'] == e_name].copy()

        # Apply gender filter
        if mode == 'Men Only':
            vdf = vdf[vdf['Gender'].isin(['M', 'MEN', 'MALE'])]
        elif mode == 'Women Only':
            vdf = vdf[vdf['Gender'].isin(['W', 'WOMEN', 'FEMALE'])]

        vdf = vdf.dropna(subset=['NumericResult'])

        if vdf.empty:
            st.warning(f"No valid numeric data found for {e_name} in {mode} mode.")
        else:
            # 1. Logic for Units and Best Achievement
            is_high = any(x in e_name for x in ['Throw', 'Jump', 'Vault', 'athlon'])
            base_date = pd.Timestamp("1970-01-01")

            unit_title = "Result"
            tick_format = None
            use_time_axis = False

            max_val = vdf['NumericResult'].max()
            if is_high:
                unit_title = "Points" if "athlon" in e_name else "Meters"
            else:
                if max_val > 100:
                    unit_title = "Time"
                    tick_format = "%M:%S"
                    use_time_axis = True
                else:
                    unit_title = "Seconds"

            # --- GRAPH 1: YEARLY TREND (Olympic Cycles) ---
            st.subheader(f"📈 Yearly Best Performance: {e_name}")

            if is_high:
                yearly_best = vdf.groupby(['Year', 'Gender'])['NumericResult'].max().reset_index()
                global_best_idx = yearly_best['NumericResult'].idxmax()
            else:
                yearly_best = vdf.groupby(['Year', 'Gender'])['NumericResult'].min().reset_index()
                global_best_idx = yearly_best['NumericResult'].idxmin()

            yearly_best = yearly_best.sort_values('Year')

            y_col = 'NumericResult'
            if use_time_axis:
                yearly_best['Time_Axis'] = yearly_best['NumericResult'].apply(
                    lambda x: base_date + pd.Timedelta(seconds=x))
                y_col = 'Time_Axis'

            fig_line = px.line(
                yearly_best, x='Year', y=y_col, color='Gender', markers=True,
                title="Olympic Peak Performance Over Time",
                color_discrete_map={'M': '#1f77b4', 'W': '#e377c2'}
            )

            # Connect gaps to keep the line continuous through missing years
            fig_line.update_traces(connectgaps=True)

            # Highlight All-Time Best
            best_row = yearly_best.loc[global_best_idx]
            fig_line.add_scatter(
                x=[best_row['Year']], y=[best_row[y_col]],
                mode='markers',
                marker=dict(color='gold', size=15, symbol='star', line=dict(width=2, color='black')),
                name='All-Time Olympic Best', showlegend=True
            )

            # Format X-axis for 4-year Olympic cycles and horizontal labels
            min_year = int(yearly_best['Year'].min())
            max_year = int(yearly_best['Year'].max())
            # Create ticks every 4 years (e.g., 1896, 1900, 1904...)
            olympic_cycle_ticks = list(range(min_year, max_year + 1, 4))

            fig_line.update_xaxes(
                type='linear',
                tickmode='array',
                tickvals=olympic_cycle_ticks,
                tickangle=0,  # Horizontal labels
                title="Year"
            )

            # Invert Y-axis for track events
            fig_line.update_layout(yaxis_title=unit_title)
            if not is_high:
                fig_line.update_yaxes(autorange="reversed")

            if tick_format:
                fig_line.update_layout(yaxis=dict(tickformat=tick_format))

            st.plotly_chart(fig_line, use_container_width=True)

            st.divider()

            # --- GRAPH 2: BEST PER COUNTRY ---
            st.subheader("🌍 Best Result per Country")

            vdf['Has_Both'] = vdf['Country'].map(vdf.groupby('Country')['Gender'].nunique()) == 2
            best_per_country = vdf.sort_values('NumericResult', ascending=not is_high).groupby(
                ['Country', 'Gender']).first().reset_index()

            if is_high:
                global_best_val = best_per_country['NumericResult'].max()
            else:
                global_best_val = best_per_country['NumericResult'].min()

            ranks = best_per_country.groupby('Country').agg(
                {'NumericResult': 'max' if is_high else 'min', 'Has_Both': 'first'}).reset_index()
            ranks = ranks.sort_values(['Has_Both', 'NumericResult'], ascending=[True, True if is_high else False])

            x_col = "NumericResult"
            if use_time_axis:
                best_per_country['Time_Axis'] = best_per_country['NumericResult'].apply(
                    lambda x: base_date + pd.Timedelta(seconds=x))
                x_col = "Time_Axis"

            fig_scatter = px.scatter(
                best_per_country, x=x_col, y="Country", color="Gender",
                color_discrete_map={'M': '#1f77b4', 'W': '#e377c2'},
                height=max(500, len(best_per_country) * 20),
                hover_name="Country",
                hover_data={'Year': True, 'Name': True, x_col: False, 'NumericResult': True}
            )

            # Highlight All-Time Best in Scatter
            best_scatter_rows = best_per_country[best_per_country['NumericResult'] == global_best_val]
            fig_scatter.add_scatter(
                x=best_scatter_rows[x_col], y=best_scatter_rows['Country'],
                mode='markers',
                marker=dict(color='gold', size=12, symbol='star', line=dict(width=1, color='black')),
                name='All-Time Olympic Best', showlegend=True
            )

            # Invert X-axis for track events
            fig_scatter.update_layout(
                yaxis=dict(categoryorder='array', categoryarray=ranks['Country'].tolist()),
                xaxis_title=unit_title
            )
            if not is_high:
                fig_scatter.update_xaxes(autorange="reversed")

            if tick_format:
                fig_scatter.update_xaxes(tickformat=tick_format)

            st.plotly_chart(fig_scatter, use_container_width=True)