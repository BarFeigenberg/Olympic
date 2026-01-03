import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def show_global_overview(data, selected_country_session):
    st.title("🏅 Global Olympic Insights")
    st.divider()

    medals_only = data[data['Medal'] != 'No medal']
    map_data = medals_only.groupby('Team')['Medal'].count().reset_index()
    map_data.rename(columns={'Medal': 'Total Medals'}, inplace=True)
    country_list = sorted(map_data['Team'].dropna().unique())
    
    col_select, col_metrics = st.columns([1, 3], gap="medium")
    with col_select:
        default_selection = st.session_state.selected_country if st.session_state.selected_country in country_list else (
            country_list[0] if country_list else "USA")

        selected_country = st.selectbox(
            "Select Country:",
            country_list,
            index=country_list.index(default_selection)
        )
        if selected_country != st.session_state.selected_country:
            st.session_state.selected_country = selected_country
            st.rerun()

    with col_metrics:
        country_df = medals_only[medals_only['Team'] == st.session_state.selected_country]
        total = len(country_df)
        best_sport = country_df['Sport'].mode()[0] if not country_df.empty else "N/A"
        best_year = country_df['Year'].mode()[0] if not country_df.empty else "N/A"

        m1, m2, m3 = st.columns(3)
        m1.metric("🥇 Total Medals", total)
        m2.metric("🏆 Best Sport", best_sport)
        m3.metric("📅 Best Year", str(best_year))

    st.divider()

    pastel_scale = [[0.0, "#C1E1C1"], [0.5, "#FDFD96"], [1.0, "#FF6961"]]
    fig_map = px.choropleth(
        map_data,
        locations="Team",
        locationmode="country names",
        color="Total Medals",
        hover_name="Team",
        custom_data=["Team"],
        color_continuous_scale=pastel_scale,
        projection="natural earth"
    )

    sel_data = map_data[map_data['Team'] == st.session_state.selected_country]
    if not sel_data.empty:
        fig_map.add_trace(go.Choropleth(
            locations=sel_data['Team'],
            locationmode="country names",
            z=[1], colorscale=[[0, 'black'], [1, 'black']],
            showscale=False, hoverinfo="skip", marker_line_color='black', marker_line_width=2
        ))

    fig_map.update_geos(showocean=True, oceancolor="#E0F7FA", showland=True, landcolor="#F5F5F5")
    fig_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=500, clickmode="event", dragmode="pan")

    map_sel = st.plotly_chart(fig_map, use_container_width=True, on_select="rerun", key="world_map")

    if map_sel and map_sel["selection"]["points"]:
        c = map_sel["selection"]["points"][0].get("customdata", [None])[0]
        if c and c != st.session_state.selected_country:
            st.session_state.selected_country = c
            st.rerun()

    st.divider()

    c_trend, c_podium = st.columns(2, gap="large")

    with c_trend:
        st.subheader("📈 Medals Trend")
        td = country_df.groupby('Year')['Medal'].count().reset_index()
        if not td.empty:
            # CHANGED: px.area -> px.line to remove filled area
            fig = px.line(td, x='Year', y='Medal', markers=True, color_discrete_sequence=["#77DD77"])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with c_podium:
        st.subheader("🏆 Top 3 Sports Podium")
        top = country_df.groupby('Sport')['Medal'].count().reset_index().sort_values('Medal', ascending=False).head(
            3).reset_index(drop=True)
        if len(top) >= 1:
            data = [{'Sport': top.iloc[0]['Sport'], 'Medal': top.iloc[0]['Medal'], 'Color': '#FFD700', 'Pos': 2}]
            if len(top) >= 2: data.append(
                {'Sport': top.iloc[1]['Sport'], 'Medal': top.iloc[1]['Medal'], 'Color': '#C0C0C0', 'Pos': 1})
            if len(top) >= 3: data.append(
                {'Sport': top.iloc[2]['Sport'], 'Medal': top.iloc[2]['Medal'], 'Color': '#CD7F32', 'Pos': 3})

            pdf = pd.DataFrame(data).sort_values('Pos')
            fig = go.Figure(go.Bar(x=pdf['Sport'], y=pdf['Medal'], marker_color=pdf['Color'], text=pdf['Medal'],
                                   textposition='auto'))
            fig.update_layout(height=400, yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)