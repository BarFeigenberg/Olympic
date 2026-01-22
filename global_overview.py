import plotly.graph_objects as go
from data_processor import *


def show_global_overview(medals_only_unused, total_medals_per_country_unused, country_list, medals_data_unused):
    """
    Display global Olympic insights:
    - Top KPIs for selected country
    - Interactive world map showing medal distribution
    - Historical medal trends over time

    @param medals_only_unused: Not used; placeholder for backward compatibility
    @param total_medals_per_country_unused: Not used; placeholder
    @param country_list: List of countries for selector
    @param medals_data_unused: Not used; placeholder
    """
    # --- 1. Load core processed datasets ---
    df_all = get_processed_medals_data_by_score_and_type()  # Full medal dataset with scores
    country_ref = get_processed_country_data()  # Mapping NOC <-> Country
    pop_df = get_combined_population_data()  # Population reference for normalization

    # --- 2. Page header ---
    st.title("Global Olympic Insights")
    st.divider()

    # --- 3. Country Selector & KPI Cards ---
    all_countries_base = get_all_world_countries()
    available_countries = sorted(all_countries_base)

    col_select, col_kpis = st.columns([1, 3], gap="medium")

    # --- Country selection logic with session state ---
    with col_select:
        if 'selected_country' not in st.session_state or st.session_state.selected_country not in available_countries:
            st.session_state.selected_country = "United States" if "United States" in available_countries else \
            available_countries[0]

        selected_country = st.selectbox(
            "Select Country:",
            available_countries,
            index=available_countries.index(st.session_state.selected_country)
        )

        if selected_country != st.session_state.selected_country:
            st.session_state.selected_country = selected_country
            st.rerun()

    # --- KPI Card Metrics ---
    current_metric_label = st.session_state.get("global_metric_radio", "Total Medals")
    metric_map = {"Total Medals": "total_count", "Weighted Score": "score", "Gold Medals": "Gold"}
    metric_col = metric_map[current_metric_label]
    metric_display_name = current_metric_label.replace(" Medals", "")

    with col_kpis:
        # Filter dataset for selected country using NOC mapping
        c_hist = df_all[df_all['noc'].isin(
            country_ref[country_ref['country'] == st.session_state.selected_country]['noc']
        )].copy()

        # Aggregate KPIs
        country_val = c_hist[metric_col].sum() if not c_hist.empty else 0

        if not c_hist.empty:
            # Best Year calculation
            year_stats = c_hist.groupby('year')[metric_col].sum().sort_values(ascending=False)
            best_year = year_stats.index[0]
            best_val = year_stats.iloc[0]

            # Top 3 sports by metric
            sport_stats = c_hist.groupby('sport')[metric_col].sum().sort_values(ascending=False).head(3)
            medals_info = [("🥇", "#FFD700"), ("🥈", "#A8A8A8"), ("🥉", "#CD7F32")]
            sport_parts = []
            for i, (sp, val) in enumerate(sport_stats.items()):
                emoji, color = medals_info[i]
                sport_parts.append(f"<span style='color:{color}; font-weight:bold;'>{emoji} {sp}</span>")
            top_sports_html = " • ".join(sport_parts) if sport_parts else "N/A"
        else:
            best_year, best_val, top_sports_html = "N/A", 0, "N/A"

        # --- Display KPI cards ---
        k1, k2, k3 = st.columns([1, 1, 1.5])
        card_style = "background-color: #ffffff; border: 1px solid #f0f2f6; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); height: 100px; display: flex; flex-direction: column; justify-content: center; text-align: center;"
        label_style = "font-size: 15px; color: #666; margin-bottom: 2px; width: 100%;"
        value_style = "font-size: 26px; font-weight: bold; color: #333; width: 100%;"

        with k1:
            st.markdown(
                f'<div style="{card_style}"><div style="{label_style}">{metric_display_name}</div><div style="{value_style}">{int(country_val):,}</div></div>',
                unsafe_allow_html=True
            )
        with k2:
            st.markdown(
                f'<div style="{card_style}"><div style="{label_style}">Best Year</div><div style="{value_style}">{best_year}</div><div style="font-size:12px; color:#999">({int(best_val)} {metric_display_name})</div></div>',
                unsafe_allow_html=True
            )
        with k3:
            st.markdown(
                f'<div style="{card_style}"><div style="{label_style}">Top Sports ({metric_display_name})</div><div style="font-size: 16px; margin-top: 5px;">{top_sports_html}</div></div>',
                unsafe_allow_html=True
            )

    st.divider()

    # --- 4. Global Map Section ---
    st.subheader("Global Map")

    # Columns for metric and normalization selection
    col_view_radio, col_metric_radio = st.columns([1, 1])

    with col_metric_radio:
        global_metric_label = st.radio(
            "Select Global Metric:",
            ["Total Medals", "Weighted Score", "Gold Medals"],
            horizontal=True,
            key="global_metric_radio"
        )

    with col_view_radio:
        map_view = st.radio(
            "Normalization:",
            ["Per Million People", "Absolute Values"],
            horizontal=True,
            key="map_view_toggle"
        )

    # Update metric selection
    metric_col = metric_map[global_metric_label]
    metric_type = "Total"
    if "Score" in global_metric_label:
        metric_type = "Score"
    elif "Gold" in global_metric_label:
        metric_type = "Gold"

    # --- 5. Prepare Map Data ---
    # Aggregate by NOC and map to country names
    noc_agg = df_all.groupby('noc')[metric_col].sum().reset_index().rename(columns={metric_col: 'value'})
    winners_named = noc_agg.merge(country_ref[['noc', 'country']], on='noc', how='left')
    winners_named['country'] = winners_named['country'].fillna(winners_named['noc'])
    winners_grouped = winners_named.groupby('country')['value'].sum().reset_index()

    # Merge with all countries
    map_data = pd.DataFrame({'country': all_countries_base}).merge(winners_grouped, on='country', how='left')
    map_data['value'] = map_data['value'].fillna(0).astype(int)

    # Merge with latest population for normalization
    latest_pop = pop_df.sort_values('year').groupby('country').tail(1)[['country', 'population']]
    map_data = map_data.merge(latest_pop, on='country', how='left')

    # Calculate per-million values
    map_data['medals_per_million'] = (map_data['value'] / map_data['population']) * 1_000_000
    map_data['medals_per_million'] = map_data['medals_per_million'].fillna(0)

    # --- Define bins for choropleth ---
    def get_labels(bins_list):
        return [f"{bins_list[i]} – {bins_list[i + 1]}" for i in range(len(bins_list) - 1)]

    if map_view == "Absolute Values":
        val_col = 'value'
        if metric_type == "Score":
            bins = [0, 20, 100, 300, 1000, 2000, 5000, 10000, 50000]
        elif metric_type == "Gold":
            bins = [0, 5, 20, 50, 100, 300, 700, 1500, 5000]
        else:
            bins = [0, 10, 50, 100, 300, 700, 1500, 3000, 10000]
    else:
        val_col = 'medals_per_million'
        if metric_type == "Score":
            bins = [0, 1, 3, 10, 25, 50, 100, 200, 1000]
        elif metric_type == "Gold":
            bins = [0, 0.1, 0.5, 1, 3, 7, 15, 30, 100]
        else:
            bins = [0, 0.2, 0.5, 1, 3, 7, 15, 30, 200]

    # Assign bin labels for visualization
    bin_labels = get_labels(bins[:-1]) + [f"{bins[-2]}+"]
    map_data['bin_label'] = pd.cut(
        map_data[val_col], bins=bins, include_lowest=True, right=False,
        labels=bin_labels
    ).astype(str)

    # --- 6. Plot Choropleth Map ---
    fig_map = px.choropleth(
        map_data[map_data['bin_label'] != 'nan'],
        locations="country",
        locationmode="country names", color="bin_label",
        hover_name="country",
        hover_data={"bin_label": False, "country": False, "value": ":,", "population": ":,", "medals_per_million": ":.2f"},
        color_discrete_sequence=["#d8f3dc","#b7e4c7","#95d5b2","#74c69d","#52b788","#40916c","#2d6a4f","#1b4332"],
        category_orders={"bin_label": bin_labels},
        projection="natural earth"
    )

    # Highlight selected country
    sel_data = map_data[map_data['country'] == st.session_state.selected_country]
    if not sel_data.empty:
        fig_map.add_trace(go.Choropleth(
            locations=sel_data['country'], locationmode="country names", z=[1],
            colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']], showscale=False,
            hoverinfo="skip", marker_line_color='black', marker_line_width=3
        ))

    fig_map.update_geos(
        showocean=True, oceancolor="#E0F7FA",
        showland=True, landcolor="#F5F5F5",
        showcountries=True, countrycolor="#AAAAAA"
    )
    fig_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=500, clickmode="event", dragmode="pan")

    # Display map in Streamlit
    map_sel = st.plotly_chart(fig_map, width='stretch', on_select="rerun", key="world_map")
    if map_sel and map_sel["selection"]["points"]:
        p = map_sel["selection"]["points"][0]
        if "location" in p and p["location"] != st.session_state.selected_country:
            st.session_state.selected_country = p["location"]
            st.rerun()

    st.divider()

    # --- 7. Historical Trend Section ---
    st.subheader(f"Historical Trend: {st.session_state.selected_country}")
    trend_norm = st.radio("Normalization:", ["Per Million People", "Absolute Count"], horizontal=True,
                          key="trend_norm_sel")

    if c_hist.empty:
        st.info("No historical data available for this country.")
    else:
        trend_df = c_hist.groupby('year')[['total_count', 'Gold', 'Silver', 'Bronze']].sum().reset_index()
        trend_df = trend_df[trend_df['year'] != 1906]  # Exclude special games
        country_pop = pop_df[pop_df['country'] == st.session_state.selected_country].copy()
        trend_df = trend_df.merge(country_pop[['year', 'population']], on='year', how='left')

        metrics = ['total_count', 'Gold', 'Silver', 'Bronze']
        if trend_norm == "Per Million People":
            for m in metrics:
                trend_df[m] = (trend_df[m] / trend_df['population']) * 1_000_000
            y_title = "Medals / Million"
        else:
            y_title = "Medals Count"

        trend_df.rename(columns={'total_count': 'Total'}, inplace=True)
        trend_long = trend_df.melt(
            id_vars='year', value_vars=['Total', 'Gold', 'Silver', 'Bronze'],
            var_name='Medal Type', value_name='Count'
        )

        fig_trend = px.line(
            trend_long, x='year', y='Count', color='Medal Type', markers=False,
            color_discrete_map={"Total": "#95d5b2", "Gold": "#DAA520", "Silver": "#A8A8A8", "Bronze": "#cc5803"}
        )
        fig_trend.update_layout(
            height=500, yaxis_title=y_title, xaxis_title="Year",
            xaxis=dict(tickmode='linear', tick0=1896, dtick=4), hovermode="x unified"
        )
        st.plotly_chart(fig_trend, width='stretch')
