import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import streamlit as st
from data_processor import *


def show_global_overview(medals_only_unused, total_medals_per_country_unused, country_list, medals_data_unused):
    """
    Refactored Global Overview with:
    1. Global Metric Selector (Total / Score / Gold) -> Updates Map & KPI
    2. Trend Chart Filters (Medal Type + Normalization)
    """
    
    # --- 1. HEADER & GLOBAL CONTROLS ---
    col_title, col_metric = st.columns([3, 1])
    with col_title:
        st.title("Global Olympic Insights")
    
    with col_metric:
        # Global Metric Selector
        global_metric_label = st.selectbox("Global Metric:", ["Total Medals", "Weighted Score", "Gold Medals"])

    st.divider()

    # Map Selection to Column Name
    # "Total Medals" in our deduplicated Event file is 'total_count' (1 per event) matches 'Events'
    # 'score' is Weighted
    # 'Gold' is Gold count
    metric_map = {
        "Total Medals": "total_count",
        "Weighted Score": "score",
        "Gold Medals": "Gold"
    }
    metric_col = metric_map[global_metric_label]

    # --- 2. DATA LOADING & PREPARATION ---
    # Use the robust, unified data source (Event-Level)
    df_all = get_processed_medals_data_by_score_and_type()
    
    # Prepare Map Data
    # 1. Aggregate Winners by NOC
    noc_agg = df_all.groupby('noc')[metric_col].sum().reset_index()
    noc_agg.rename(columns={metric_col: 'value'}, inplace=True)
    
    # 2. Map NOC to Country Name
    country_ref = get_processed_country_data()
    winners_named = noc_agg.merge(country_ref[['noc', 'country']], on='noc', how='left')
    winners_named['country'] = winners_named['country'].fillna(winners_named['noc'])
    
    # 3. Group by Country Name (handles rare multi-NOC mapping)
    winners_grouped = winners_named.groupby('country')['value'].sum().reset_index()

    # 4. Merge with ALL World Countries (to ensure 0-medal countries appear)
    all_countries = get_all_world_countries()
    map_data = pd.DataFrame({'country': all_countries})
    map_data = map_data.merge(winners_grouped, on='country', how='left')
    
    # 5. Fill Missing with 0
    map_data['value'] = map_data['value'].fillna(0).astype(int)
    
    # Merge with Population for Per Capita calculation
    pop_df = get_combined_population_data()
    latest_pop = pop_df.sort_values('year').groupby('country').tail(1)[['country', 'population']]
    map_data = map_data.merge(latest_pop, on='country', how='left')
    
    # Calculate Per Million for Map Hover/Logic
    map_data['medals_per_million'] = (map_data['value'] / map_data['population']) * 1_000_000
    map_data['medals_per_million'] = map_data['medals_per_million'].fillna(0)
    
    # --- 3. COUNTRY SELECTOR (Row 1) ---
    col_select, col_kpis = st.columns([1, 3], gap="medium")

    with col_select:
        # Ensure 'country_list' is valid (derived from df_all could be better, but passed ref is ok)
        # Let's derive from df_all to be consistent with the metric data
        available_countries = sorted(map_data['country'].unique())
        
        if 'selected_country' not in st.session_state or st.session_state.selected_country not in available_countries:
            # Default to USA or first
            st.session_state.selected_country = "United States" if "United States" in available_countries else (available_countries[0] if available_countries else "")

        selected_country = st.selectbox(
            "Select Country:",
            available_countries,
            index=available_countries.index(st.session_state.selected_country) if st.session_state.selected_country in available_countries else 0
        )

        if selected_country != st.session_state.selected_country:
            st.session_state.selected_country = selected_country
            st.rerun()

    # --- 4. KPI CARDS (Row 1 Right) - BASED ON GLOBAL METRIC ---
    with col_kpis:
        # Filter for selected country
        country_row = map_data[map_data['country'] == st.session_state.selected_country]
        if not country_row.empty:
            total_val = country_row.iloc[0]['value']
            # rank calculation?
            # map_data sorted
            rank_df = map_data.sort_values('value', ascending=False).reset_index(drop=True)
            rank = rank_df[rank_df['country'] == st.session_state.selected_country].index[0] + 1
        else:
            total_val = 0
            rank = "-"

        # Best Year (based on selected metric)
        c_hist = df_all[df_all['noc'].isin(country_ref[country_ref['country'] == st.session_state.selected_country]['noc'])].copy()
        if not c_hist.empty:
             year_stats = c_hist.groupby('year')[metric_col].sum().sort_values(ascending=False)
             best_year = year_stats.index[0]
             best_val = year_stats.iloc[0]
        else:
             best_year = "N/A"
             best_val = 0

        # Top Sport (based on selected metric)
        if not c_hist.empty and 'sport' in c_hist.columns:
             sport_stats = c_hist.groupby('sport')[metric_col].sum().sort_values(ascending=False).head(1)
             if not sport_stats.empty:
                 top_sport = f"{sport_stats.index[0]} ({int(sport_stats.iloc[0])})"
             else:
                 top_sport = "N/A"
             
             # Also get Top 3 for display
             top_3_sports = c_hist.groupby('sport')[metric_col].sum().sort_values(ascending=False).head(3).index.tolist()
             # Formatting
             medals_info = [("🥇", "#FFD700"), ("🥈", "#A8A8A8"), ("🥉", "#CD7F32")]
             sport_parts = []
             for i, sp in enumerate(top_3_sports):
                 emoji, color = medals_info[i]
                 sport_parts.append(f"<span style='color:{color}; font-weight:bold;'>{emoji} {sp}</span>")
             top_sports_html = " • ".join(sport_parts) if sport_parts else "N/A"
        else:
             top_sports_html = "N/A"

        # Display KPIs
        k1, k2, k3 = st.columns([1, 1, 1.5])
        
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
                text-align: center;
            """
        label_style = "font-size: 15px; color: #666; margin-bottom: 2px; width: 100%;"
        value_style = "font-size: 26px; font-weight: bold; color: #333; width: 100%;"

        metric_display_name = global_metric_label.replace(" Medals", "") # "Total", "Weighted Score", "Gold"

        with k1:
            st.markdown(f"""
                    <div style="{card_style}">
                        <div style="{label_style}">{metric_display_name} (Rank #{rank})</div>
                        <div style="{value_style}">{int(total_val):,}</div>
                    </div>
                """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
                    <div style="{card_style}">
                        <div style="{label_style}">Best Year</div>
                        <div style="{value_style}">{best_year}</div>
                        <div style="font-size:12px; color:#999">({int(best_val)} {metric_display_name})</div>
                    </div>
                """, unsafe_allow_html=True)
        with k3:
             st.markdown(f"""
                    <div style="{card_style}">
                        <div style="{label_style}">Top Sports ({metric_display_name})</div>
                        <div style="font-size: 16px; margin-top: 5px;">{top_sports_html}</div>
                    </div>
                """, unsafe_allow_html=True)

    st.divider()

    # --- ROW 2: WORLD MAP ---
    c_map_title, c_map_ctrl = st.columns([1, 1])
    with c_map_title:
        st.subheader(f"Global Map: {global_metric_label}")
    with c_map_ctrl:
        # Local toggle for Map Normalization
        map_view = st.radio("View:", ["Absolute Values", "Per Million People"], horizontal=True, key="map_view_toggle")

    # --- BINNING LOGIC ---
    # We define specific bins for each combination to ensure good color spread
    # Metric: "Total", "Score", "Gold" (derived from global_metric_label)
    # Mode: "Absolute", "Per Million"
    
    metric_type = "Total"
    if "Score" in global_metric_label: metric_type = "Score"
    elif "Gold" in global_metric_label: metric_type = "Gold"
    
    # helper to format labels
    def get_labels(bins_list):
        return [f"{bins_list[i]} – {bins_list[i+1]}" for i in range(len(bins_list)-1)]

    if map_view == "Absolute Values":
        val_col = 'value'
        if metric_type == "Score":
             # Score is approx 2-3x Total. Ranges scaled up.
             bins = [0, 20, 100, 300, 1000, 2000, 5000, 10000, 50000]
             discrete_reds = px.colors.sequential.Reds[1:9]
        elif metric_type == "Gold":
             # Gold is approx 1/3 Total. Ranges scaled down.
             bins = [0, 5, 20, 50, 100, 300, 700, 1500, 5000]
             discrete_reds = px.colors.sequential.Reds[1:9]
        else:
             # Total (Original Reference)
             bins = [0, 10, 50, 100, 300, 700, 1500, 3000, 10000]
             discrete_reds = px.colors.sequential.Reds[1:9]
    else:
        val_col = 'medals_per_million'
        if metric_type == "Score":
             # Score per Million
             bins = [0, 1, 3, 10, 25, 50, 100, 200, 1000]
             discrete_reds = px.colors.sequential.Reds[1:9]
        elif metric_type == "Gold":
             # Gold per Million
             bins = [0, 0.1, 0.5, 1, 3, 7, 15, 30, 100]
             discrete_reds = px.colors.sequential.Reds[1:9]
        else:
             # Total per Million (Original Reference)
             bins = [0, 0.2, 0.5, 1, 3, 7, 15, 30, 200]
             discrete_reds = px.colors.sequential.Reds[1:9]

    bin_labels = get_labels(bins[:-1]) + [f"{bins[-2]}+"]
    
    # Apply Binning
    # Handle infinite or very large values safely
    map_data['bin_label'] = pd.cut(
        map_data[val_col],
        bins=bins,
        include_lowest=True,
        right=False,
        labels=bin_labels
    ).astype(str)
    
    # Filter 'nan' bins (meaning out of range or no data, usually 0 is covered)
    map_viz_df = map_data[map_data['bin_label'] != 'nan'].copy()
    
    # Ensure color order respects the bin order
    fig_map = px.choropleth(
        map_viz_df,
        locations="country",
        locationmode="country names",
        color="bin_label",
        hover_name="country",
        hover_data={"bin_label": False, "country": False, "value": True, "population": True, "medals_per_million": ":.2f"},
        color_discrete_sequence=discrete_reds,
        category_orders={"bin_label": bin_labels},
        projection="natural earth",
        labels={"value": f"Total ({metric_type})", "medals_per_million": f"{metric_type}/1M", "bin_label": "Range"}
    )

    # Highlight Selection
    sel_data = map_data[map_data['country'] == st.session_state.selected_country]
    if not sel_data.empty:
        fig_map.add_trace(go.Choropleth(
            locations=sel_data['country'],
            locationmode="country names",
            z=[1],
            colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']],
            showscale=False,
            hoverinfo="skip",
            marker_line_color='black', marker_line_width=3
        ))

    fig_map.update_geos(showocean=True, oceancolor="#E0F7FA", showland=True, landcolor="#F5F5F5", showcountries=True, countrycolor="#AAAAAA")
    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=500, clickmode="event", dragmode="pan")

    map_sel = st.plotly_chart(fig_map, width='stretch', on_select="rerun", key="world_map")
    if map_sel and map_sel["selection"]["points"]:
        p = map_sel["selection"]["points"][0]
        if "location" in p:
            c = p["location"]
            if c != st.session_state.selected_country:
                st.session_state.selected_country = c
                st.rerun()

    st.divider()

    # --- ROW 3: MEDALS TREND (Advanced Filters) ---
    st.subheader(f"Historical Trend: {st.session_state.selected_country}")
    
    # Advanced Controls
    t_ctrl1, t_ctrl2 = st.columns([1, 1])
    
    with t_ctrl1:
        # Filter 1: Medal Type for the Chart
        trend_medal = st.radio("Chart Metric:", ["Total", "Gold", "Silver", "Bronze"], horizontal=True, key="trend_medal_sel")
    with t_ctrl2:
        # Filter 2: Normalization
        trend_norm = st.radio("Normalization:", ["Absolute Count", "Per Million People"], horizontal=True, key="trend_norm_sel")

    # Prepare Trend Data
    # Filter for country
    # We used 'c_hist' earlier (Filtered by Country)
    # c_hist = df_all[ ... selected_country ]
    
    if c_hist.empty:
        st.info("No historical data available for this country.")
    else:
        # 1. Select the Value Column based on 'trend_medal'
        # Note: df_all (Detailed) has 'Gold', 'Silver', 'Bronze' columns which are 0/1 for events, 
        # or we can count rows.
        # But 'Score' logic is complex here.
        # If user selects "Total" -> Count events (sum 'total_count')
        # If "Gold" -> Sum 'Gold' col
        # If "Silver" -> Sum 'Silver' col
        # If "Bronze" -> Sum 'Bronze' col
        
        target_col_map = {
            "Total": "total_count",
            "Gold": "Gold",
            "Silver": "Silver",
            "Bronze": "Bronze"
        }
        y_raw_col = target_col_map[trend_medal]
        
        # Group by Year
        trend_df = c_hist.groupby('year')[y_raw_col].sum().reset_index()
        trend_df.rename(columns={y_raw_col: 'value'}, inplace=True)
        
        # Merge Population for Per Million
        country_pop = pop_df[pop_df['country'] == st.session_state.selected_country].copy()
        trend_df = trend_df.merge(country_pop[['year', 'population']], on='year', how='left')
        
        # Calculate Final Y
        if trend_norm == "Per Million People":
            trend_df['plot_value'] = (trend_df['value'] / trend_df['population']) * 1_000_000
            y_title = f"{trend_medal} Medals / Million"
            tooltip_key = 'Per Million'
        else:
            trend_df['plot_value'] = trend_df['value']
            y_title = f"{trend_medal} Medals (Count)"
            tooltip_key = 'Count'
            
        # Plot
        # Colors based on Medal Type
        color_map = {
            "Total": "#DC143C",   # Red
            "Gold": "#DAA520",    # Gold
            "Silver": "#A8A8A8",  # Silver
            "Bronze": "#CD7F32"   # Bronze
        }
        
        fig_trend = px.line(
            trend_df, x='year', y='plot_value', markers=True,
            title=f"{trend_medal} Medals over Time ({trend_norm})",
            color_discrete_sequence=[color_map[trend_medal]]
        )
        fig_trend.update_layout(
            height=450,
            yaxis_title=y_title,
            xaxis_title="Year",
            xaxis=dict(tickmode='linear', tick0=1896, dtick=4)
        )
        st.plotly_chart(fig_trend, width='stretch')
