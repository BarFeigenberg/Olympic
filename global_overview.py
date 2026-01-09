import plotly.graph_objects as go
from app import *


def show_global_overview(data, selected_country_session):
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
        sport_medals_df = medals_only.groupby(['Year', 'NOC', 'Sport']).size().reset_index(name='Medal')
        country_df_update = sport_medals_df.merge(get_processed_country_data(), left_on='NOC', right_on='noc',
                                                  how='left')
        country_df = country_df_update[country_df_update['country'] == st.session_state.selected_country]
        total_row = map_data[map_data['country'] == st.session_state.selected_country]
        total = int(total_row['total'].values[0])
        best_sport = country_df['Sport'].mode()[0] if not country_df.empty else "N/A"
        best_year = country_df['Year'].mode()[0] if not country_df.empty else "N/A"

        m1, m2, m3 = st.columns(3)
        m1.metric("🥇 Total Medals", total)
        m2.metric("🏆 Best Sport", best_sport)
        m3.metric("📅 Best Year", str(best_year))

    st.divider()

    # --- Data Processing for Map (Normalization) ---
    map_df = map_data.copy()

    # Load population data (using Gapminder 2007 as reference)
    gapminder = px.data.gapminder().query("year == 2007")

    # Fix country names to match Gapminder dataset
    name_fix = {
        "United States of America": "United States",
        "USA": "United States",
        "People's Republic of China": "China",
        "Great Britain": "United Kingdom",
        "Republic of Korea": "Korea, Rep.",
        "Russian Federation": "Russia",
    }
    map_df['merge_name'] = map_df['country'].replace(name_fix)

    # Merge map data with population
    map_df = map_df.merge(gapminder[['country', 'pop', 'iso_alpha']],
                          left_on='merge_name', right_on='country', how='left')

    # Manual population fix for Russia (often missing in Gapminder subset)
    map_df.loc[map_df['country_x'] == 'Russia', 'pop'] = 144_000_000

    # Calculate Medals per 1 Million people
    map_df['Medals_Per_Million'] = (map_df['total'] / map_df['pop']) * 1_000_000

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

    map_df['Bin_Label'] = pd.cut(
        map_df['Medals_Per_Million'],
        bins=bins,
        labels=bin_labels,
        include_lowest=True,
        right=False
    )
    map_df['Bin_Label'] = map_df['Bin_Label'].astype(str)

    # Use exactly 9 reds (light -> dark)
    discrete_reds = px.colors.sequential.Reds[1:10]
    map_df = map_df[map_df['Bin_Label'] != 'nan']

    # ROW 2: MAP
    fig_map = px.choropleth(
        map_df,
        locations="country_x",
        locationmode="country names",
        color="Bin_Label",
        hover_name="country_x",
        hover_data={
            "Bin_Label": False,
            "country_x": False,
            "total": True,
            "Medals_Per_Million": ":.1f"
        },
        color_discrete_sequence=discrete_reds,
        category_orders={"Bin_Label": bin_labels},
        projection="natural earth"
    )

    # ... (Rest of the layout code remains the same)
    sel_data = map_df[map_df['country_x'] == st.session_state.selected_country]
    if not sel_data.empty:
        fig_map.add_trace(go.Choropleth(
            locations=sel_data['country_x'],
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
    map_sel = st.plotly_chart(fig_map, use_container_width=True, on_select="rerun", key="world_map")

    if map_sel and map_sel["selection"]["points"]:
        c = map_sel["selection"]["points"][0].get("customdata", [None])[0]
        if c and c != st.session_state.selected_country:
            st.session_state.selected_country = c
            st.rerun()

    st.divider()

    # ROW 3: GRAPHS
    c_trend, c_podium = st.columns(2, gap="large")
    with c_trend:
        st.subheader("📈 Medals Trend")
        td = country_df.groupby('Year')['Medal'].count().reset_index()
        if not td.empty:
            fig = px.line(td, x='Year', y='Medal', markers=True, color_discrete_sequence=["#77DD77"])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with c_podium:
        st.subheader("🏆 Top 3 Sports Podium")
        top = country_df.groupby('Sport')['Medal'].count().reset_index().sort_values('Medal', ascending=False).head(
            3).reset_index(drop=True)
        if len(top) >= 1:
            pod_data = [{'Sport': top.iloc[0]['Sport'], 'Medal': top.iloc[0]['Medal'], 'Color': '#FFD700', 'Pos': 2}]
            if len(top) >= 2: pod_data.append(
                {'Sport': top.iloc[1]['Sport'], 'Medal': top.iloc[1]['Medal'], 'Color': '#C0C0C0', 'Pos': 1})
            if len(top) >= 3: pod_data.append(
                {'Sport': top.iloc[2]['Sport'], 'Medal': top.iloc[2]['Medal'], 'Color': '#CD7F32', 'Pos': 3})
            pdf = pd.DataFrame(pod_data).sort_values('Pos')
            fig = go.Figure(go.Bar(x=pdf['Sport'], y=pdf['Medal'], marker_color=pdf['Color'], text=pdf['Medal'],
                                   textposition='auto'))
            fig.update_layout(height=400, yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)



