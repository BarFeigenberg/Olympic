import plotly.graph_objects as go
from data_processor import *


def show_global_overview(medals_only, total_medals_per_country, country_list, medals_data):
    st.title("Global Olympic Insights")
    st.divider()

    # --- 1. Data Preparation for Map & Global Stats ---
    pop_df = get_combined_population_data()
    latest_pop = pop_df.sort_values('year').groupby('country').tail(1)[['country', 'population']]

    # Prepare Map Data - start with ALL countries from continent_data.csv
    all_countries = get_all_world_countries()
    all_countries_df = pd.DataFrame({'country': all_countries})
    map_df = all_countries_df.merge(total_medals_per_country, on='country', how='left')

    # Fill missing values for countries without medals
    map_df['total'] = map_df['total'].fillna(0).astype(int)
    if 'medals' not in map_df.columns:
        map_df['medals'] = map_df['total']
    map_df['medals'] = map_df['medals'].fillna(0).astype(int)

    # Merge with population
    map_df = map_df.merge(latest_pop, on='country', how='left')
    map_df = calculate_medals_per_million(map_df)
    map_df['medals_per_million'] = map_df['medals_per_million'].fillna(0)

    # --- ROW 1: CONTROLS & METRICS ---
    col_select, col_metrics = st.columns([1, 3], gap="medium")

    with col_select:
        if 'selected_country' not in st.session_state or st.session_state.selected_country not in country_list:
            st.session_state.selected_country = country_list[0] if country_list else "USA"

        selected_country = st.selectbox(
            "Select Country:",
            country_list,
            index=country_list.index(st.session_state.selected_country)
        )

        if selected_country != st.session_state.selected_country:
            st.session_state.selected_country = selected_country
            st.rerun()

    with col_metrics:
        # Prepare metrics data
        sport_medals_df = medals_only.groupby(['year', 'noc', 'sport']).size().reset_index(name='medal')
        country_df_update = sport_medals_df.merge(get_processed_country_data(), left_on='noc', right_on='noc',
                                                  how='left')
        country_df = country_df_update[country_df_update['country'] == st.session_state.selected_country]

        country_stats = map_df[map_df['country'] == st.session_state.selected_country]

        if not country_stats.empty:
            total = int(country_stats['total'].values[0])
        else:
            total = 0

        best_year = country_df['year'].mode()[0] if not country_df.empty else "N/A"

        # Get Top 3 Sports
        top_sports = country_df.groupby('sport')['medal'].count().reset_index().sort_values('medal',
                                                                                            ascending=False).head(
            3).reset_index(drop=True)

        # Build top sports text with medal colors
        if len(top_sports) >= 1:
            medals_info = [("🥇", "#FFD700"), ("🥈", "#A8A8A8"), ("🥉", "#CD7F32")]
            sport_parts = []
            for i in range(min(3, len(top_sports))):
                emoji, color = medals_info[i]
                sport_name = top_sports.iloc[i]['sport']
                sport_parts.append(f"<span style='color:{color}; font-weight:bold;'>{emoji} {sport_name}</span>")
            top_sports_html = " • ".join(sport_parts)
        else:
            top_sports_html = "N/A"

        # Display metrics in a row: Total Medals | Best Year | Top Sports
        m1, m2, m3 = st.columns([1, 1, 1.5])  # Adjusted ratios for better fit

        # Consistent style for all cards
        card_style = """
                background-color: #ffffff;
                border: 1px solid #f0f2f6;
                padding: 15px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
                height: 100px;
                display: flex;
                flex-direction: column;
                justify-content: center;
            """
        label_style = "font-size: 16px; color: rgb(49, 51, 63); margin-bottom: 2px; opacity: 0.8;"
        value_style = "font-size: 27px; font-weight: 500; color: rgb(49, 51, 63);"

        with m1:
            st.markdown(f"""
                    <div style="{card_style}">
                        <div style="{label_style}">Total Medals</div>
                        <div style="{value_style}">{total}</div>
                    </div>
                """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
                    <div style="{card_style}">
                        <div style="{label_style}">Best Year</div>
                        <div style="{value_style}">{best_year}</div>
                    </div>
                """, unsafe_allow_html=True)

        with m3:
            st.markdown(f"""
                    <div style="{card_style}">
                        <div style="{label_style}">Top Sports</div>
                        <div style="font-size: 16px; font-weight: 1800;">{top_sports_html}</div>
                    </div>
                """, unsafe_allow_html=True)

    st.divider()

    # --- ROW 2: WORLD MAP ---
    bins = [0, 0.2, 0.5, 1, 3, 7, 15, 30, 2000]
    bin_labels = ["0 – 0.2", "0.2 – 0.5", "0.5 – 1", "1 – 3", "3 – 7", "7 – 15", "15 – 30", "30 – 60"]

    if 'medals_per_million' in map_df.columns:
        map_df['bin_label'] = pd.cut(
            map_df['medals_per_million'],
            bins=bins,
            labels=bin_labels,
            include_lowest=True,
            right=False
        ).astype(str)
        map_viz_df = map_df[map_df['bin_label'] != 'nan'].copy()
    else:
        map_viz_df = map_df.copy()
        map_viz_df['bin_label'] = "0 – 0.2"

    discrete_reds = px.colors.sequential.Reds[1:10]

    fig_map = px.choropleth(
        map_viz_df,
        locations="country",
        locationmode="country names",
        color="bin_label",
        hover_name="country",
        hover_data={"bin_label": False, "country": False, "total": True, "population": True,
                    "medals_per_million": ":.2f"},
        color_discrete_sequence=discrete_reds,
        category_orders={"bin_label": bin_labels},
        projection="natural earth"
    )

    # Highlight selected country with BLACK border
    sel_data = map_viz_df[map_viz_df['country'] == st.session_state.selected_country]
    if not sel_data.empty:
        fig_map.add_trace(go.Choropleth(
            locations=sel_data['country'],
            locationmode="country names",
            z=[1],
            colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']],  # Transparent fill
            showscale=False,
            hoverinfo="skip",
            marker_line_color='black',
            marker_line_width=3
        ))

    # Show ALL countries with borders
    fig_map.update_geos(
        showocean=True, oceancolor="#E0F7FA",
        showland=True, landcolor="#F5F5F5",
        showcountries=True, countrycolor="#AAAAAA", countrywidth=0.5
    )
    fig_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=500, clickmode="event", dragmode="pan",
                          legend_title_text="Medals per 1M")

    map_sel = st.plotly_chart(fig_map, width='stretch', on_select="rerun", key="world_map")
    if map_sel and map_sel["selection"]["points"]:
        point = map_sel["selection"]["points"][0]
        if "location" in point:
            c = point["location"]
            if c and c != st.session_state.selected_country:
                st.session_state.selected_country = c
                st.rerun()

    st.divider()

    # --- ROW 3: MEDALS TREND (Full Width) ---
    t_col1, t_col2 = st.columns([2, 1])
    with t_col1:
        st.subheader("Medals Trend")
    with t_col2:
        trend_mode = st.radio("Metric:", ["Total", "Per Million"], horizontal=True, label_visibility="collapsed")

    # 1. Filter raw data
    raw_td = medals_data[medals_data['country'] == st.session_state.selected_country].copy()

    if not raw_td.empty:
        # 2. Aggregation Fix: Ensure strictly 1 row per year before merging population
        if 'total' in raw_td.columns:
            td_grouped = raw_td.groupby('year', as_index=False)['total'].sum()
            td_grouped.rename(columns={'total': 'medals'}, inplace=True)
        else:
            td_grouped = pd.DataFrame(columns=['year', 'medals'])

        # Add country column back for the merge
        td_grouped['country'] = st.session_state.selected_country

        # 3. Merge with population (Yearly basis)
        country_pop = pop_df[pop_df['country'] == st.session_state.selected_country].copy()
        td = td_grouped.merge(country_pop[['year', 'population']], on='year', how='left')

        # 4. Calculate Medals per Million
        td = calculate_medals_per_million(td)

        # Plotting Logic
        if trend_mode == "Total":
            y_val = 'medals'
            color_seq = ["#DC143C"]
            y_title = "Total Medals"
        else:
            y_val = 'medals_per_million'
            color_seq = ["#1E90FF"]
            y_title = "Medals / Million"

        if y_val in td.columns:
            fig = px.line(td, x='year', y=y_val, markers=False,
                          color_discrete_sequence=color_seq,
                          hover_data={'medals': True, 'population': True, 'medals_per_million': ':.2f'})
            fig.update_layout(height=400, yaxis_title=y_title)
            st.plotly_chart(fig, width='stretch')
        else:
            st.error(f"Missing data for {y_val}")
    else:
        st.info(f"No medal history data for {st.session_state.selected_country}")