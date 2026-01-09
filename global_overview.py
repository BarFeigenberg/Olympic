import plotly.graph_objects as go
from data_processor import *


def show_global_overview(medals_only, total_medals_per_country, country_list, medals_data):
    st.title("🏅 Global Olympic Insights")
    st.divider()

    # ROW 1: CONTROLS
    col_select, col_metrics = st.columns([1, 3], gap="medium")
    with col_select:
        if st.session_state.selected_country not in country_list:
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
        sport_medals_df = medals_only.groupby(['year', 'noc', 'sport']).size().reset_index(name='medal')
        country_df_update = sport_medals_df.merge(get_processed_country_data(), left_on='noc', right_on='noc',
                                                  how='left')
        country_df = country_df_update[country_df_update['country'] == st.session_state.selected_country]
        total_row = total_medals_per_country[total_medals_per_country['country'] == st.session_state.selected_country]
        total = int(total_row['total'].values[0])
        best_sport = country_df['sport'].mode()[0] if not country_df.empty else "N/A"
        best_year = country_df['year'].mode()[0] if not country_df.empty else "N/A"

        m1, m2, m3 = st.columns(3)
        m1.metric("🥇 Total Medals", total)
        m2.metric("🏆 Best Sport", best_sport)
        m3.metric("📅 Best Year", str(best_year))

    st.divider()

    # --- Data Processing for Map (Normalization) ---
    map_df = total_medals_per_country.copy()

    # Load population data (using Gapminder 2007 as reference)
    gapminder = px.data.gapminder().query("year == 2007")

    # Merge map data with population
    map_df = map_df.merge(gapminder[['country', 'pop', 'iso_alpha']],
                          left_on='country', right_on='country', how='left')

    # Manual population fix for Russia (often missing in Gapminder subset)
    map_df.loc[map_df['country'] == 'Russia', 'pop'] = 144_000_000

    # Calculate Medals per 1 Million people
    map_df['medals_per_million'] = (map_df['total'] / map_df['pop']) * 1_000_000

    # --- 1. Fixed, meaningful bins (9 ranges) ---
    bins = [0, 0.2, 0.5, 1, 3, 7, 15, 30, 60, 200]

    bin_labels = [
        "0 – 0.2",
        "0.2 – 0.5",
        "0.5 – 1",
        "1 – 3",
        "3 – 7",
        "7 – 15",
        "15 – 30",
        "30 – 60",
        "60+"
    ]

    map_df['bin_label'] = pd.cut(
        map_df['medals_per_million'],
        bins=bins,
        labels=bin_labels,
        include_lowest=True,
        right=False
    )
    map_df['bin_label'] = map_df['bin_label'].astype(str)

    # Use exactly 9 reds (light -> dark)
    discrete_reds = px.colors.sequential.Reds[1:10]
    map_df = map_df[map_df['bin_label'] != 'nan']

    # ROW 2: MAP
    fig_map = px.choropleth(
        map_df,
        locations="country",
        locationmode="country names",
        color="bin_label",
        hover_name="country",
        hover_data={
            "bin_label": False,
            "country": False,
            "total": True,
            "medals_per_million": ":.1f"
        },
        color_discrete_sequence=discrete_reds,
        category_orders={"bin_label": bin_labels},
        projection="natural earth"
    )

    # ... (Rest of the layout code remains the same)
    sel_data = map_df[map_df['country'] == st.session_state.selected_country]
    if not sel_data.empty:
        fig_map.add_trace(go.Choropleth(
            locations=sel_data['country'],
            locationmode="country names",
            z=[1], colorscale=[[0, 'black'], [1, 'black']],
            showscale=False, hoverinfo="skip", marker_line_color='black', marker_line_width=2
        ))

    fig_map.update_geos(showocean=True, oceancolor="#E0F7FA", showland=True, landcolor="#F5F5F5")
    fig_map.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=500,
        clickmode="event",
        dragmode="pan",
        legend_title_text="Medals per 1M People"
    )
    map_sel = st.plotly_chart(fig_map, width='stretch', on_select="rerun", key="world_map")

    if map_sel and map_sel["selection"]["points"]:
        c = map_sel["selection"]["points"][0].get("customdata", [None])[0]
        if c and c != st.session_state.selected_country:
            st.session_state.selected_country = c
            st.rerun()

    st.divider()

    # ROW 3: GRAPHS
    c_trend, c_podium = st.columns(2, gap="large")
    with c_trend:
        st.subheader("Medals Trend")
        td = medals_data[medals_data['country'] == st.session_state.selected_country]
        if not td.empty:
            fig = px.line(td, x='year', y='total', markers=True, color_discrete_sequence=["#77DD77"])
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')

    with c_podium:
        st.subheader("🏆 Top 3 Sports Podium")
        top = country_df.groupby('sport')['medal'].count().reset_index().sort_values('medal', ascending=False).head(
            3).reset_index(drop=True)
        if len(top) >= 1:
            pod_data = [{'sport': top.iloc[0]['sport'], 'medal': top.iloc[0]['medal'], 'color': '#FFD700', 'Pos': 2}]
            if len(top) >= 2: pod_data.append(
                {'sport': top.iloc[1]['sport'], 'medal': top.iloc[1]['medal'], 'color': '#C0C0C0', 'Pos': 1})
            if len(top) >= 3: pod_data.append(
                {'sport': top.iloc[2]['sport'], 'medal': top.iloc[2]['medal'], 'color': '#CD7F32', 'Pos': 3})
            pdf = pd.DataFrame(pod_data).sort_values('Pos')
            fig = go.Figure(go.Bar(x=pdf['sport'], y=pdf['medal'], marker_color=pdf['color'], text=pdf['medal'],
                                   textposition='auto'))
            fig.update_layout(height=400, yaxis_title="Count")
            st.plotly_chart(fig, width='stretch')



