import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def show_host_advantage(host_data, medals_only, country_ref):
    # --- 1. PREPARE DATA FOR DROPDOWN (Must be done before layout) ---
    noc_map = {}
    if not country_ref.empty:
        noc_map = dict(zip(country_ref['noc'], country_ref['country']))

    def get_label(row):
        full_name = noc_map.get(row['host_noc'], row['host_noc'])
        return f"{row['year']} - {row['host_city']} ({full_name})"

    host_data['label'] = host_data.apply(get_label, axis=1)
    options = sorted(host_data['label'].unique(), reverse=True)

    # --- 2. NEW LAYOUT: Title Left, Dropdown Right ---
    # Ratio [5, 2]: Title takes 5 parts, Dropdown takes 2 parts
    c_title, c_sel = st.columns([5, 2], gap="large")

    with c_title:
        st.title("🏠 The Host Effect")
        st.markdown("Does hosting the Olympics actually guarantee more medals?")

    with c_sel:
        # Add spacers to push the dropdown down to align with the title text
        st.write("")
        st.write("")
        sel_event = st.selectbox("Select Host Event:", options)

    st.divider()

    # Extract Selection Variables EARLY
    h_year = None
    h_medals = None
    h_noc = None
    full_country_name = None
    if sel_event:
        row = host_data[host_data['label'] == sel_event].iloc[0]
        h_year = int(row['year'])
        h_noc = row['host_noc']
        h_medals = int(row['total_medals'])
        full_country_name = noc_map.get(h_noc, h_noc)

    # --- 3. THE GLOBAL "BIG QUESTION" CHART ---
    st.subheader("🌍 The Big Picture: Does Hosting Pay Off?")

    # Calculate 'Lift %'
    host_data['lift_percent'] = (host_data['lift'] - 1) * 100

    # FIX: Force 1896 to 0%
    host_data.loc[host_data['year'] == 1896, 'lift_percent'] = 0
    host_data['color'] = host_data['lift_percent'].apply(lambda x: '#2ECC71' if x >= 0 else '#E74C3C')

    # Sort Data
    global_chart_data = host_data.sort_values('year').reset_index(drop=True)

    # --- TIMELINE SLIDER LOGIC ---

    # 1. Create Placeholder for Chart
    chart_place = st.empty()

    # 2. Render GREY LINE SLIDER (CSS applied above)
    all_years = sorted(global_chart_data['year'].unique())

    # Determine default value (center on selection or middle of history)
    default_year_val = all_years[len(all_years) // 2]
    if h_year and h_year in all_years:
        default_year_val = h_year

    # Use select_slider for the "timeline" look (points on a line)
    center_year = st.select_slider(
        "Timeline",  # Label hidden by CSS
        options=all_years,
        value=default_year_val
    )

    # 3. Calculate Window
    try:
        center_idx = global_chart_data[global_chart_data['year'] == center_year].index[0]
    except IndexError:
        center_idx = 0

    # Window logic: 4 before, 4 after (Total 9)
    window_size = 9
    half_window = 4

    start_idx = center_idx - half_window
    end_idx = center_idx + half_window

    # Handle edges
    if start_idx < 0:
        start_idx = 0
        end_idx = min(len(global_chart_data) - 1, window_size - 1)
    elif end_idx >= len(global_chart_data):
        end_idx = len(global_chart_data) - 1
        start_idx = max(0, end_idx - window_size + 1)

    filtered_global_data = global_chart_data.iloc[start_idx: end_idx + 1]

    # 4. Generate Chart
    fig_global = go.Figure()
    fig_global.add_trace(go.Bar(
        x=filtered_global_data['year'],
        y=filtered_global_data['lift_percent'],
        marker_color=filtered_global_data['color'],
        text=filtered_global_data['host_noc'],
        hovertemplate="<b>%{text} (%{x})</b><br>Impact: %{y:.1f}%<extra></extra>"
    ))

    fig_global.update_layout(
        yaxis_title="Performance Boost (%)",
        xaxis_title="Year",
        height=350,
        xaxis=dict(type='category', fixedrange=True),
        shapes=[dict(type="line", x0=-0.5, x1=len(filtered_global_data) - 0.5, y0=0, y1=0,
                     line=dict(color="black", width=1))]
    )

    # 5. Place Chart in placeholder (ABOVE slider)
    chart_place.plotly_chart(fig_global, width='stretch')
    st.caption("Tip: Drag the grey slider above to scroll through the Olympic history.")

    st.divider()

    # --- 4. DRILL DOWN (DEEP DIVE) ---
    if sel_event:
        st.subheader(f"🔍 Country Deep Dive: {full_country_name}")
        country_history = medals_only[medals_only['noc'] == h_noc].groupby('year')['medal'].count().reset_index()

        # KPI Calculations
        pre_years = country_history[(country_history['year'] < h_year) & (country_history['year'] >= h_year - 12)]
        avg_pre = pre_years['medal'].mean() if not pre_years.empty else 0

        diff = h_medals - avg_pre
        boost_pct = (diff / avg_pre * 100) if avg_pre > 0 else 0

        with st.container(border=True):
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("🏅 Host Year Medals", h_medals)
            k2.metric("📊 Pre-Host Avg (12y)", f"{avg_pre:.1f}")
            k3.metric("📈 Net Gain", f"+{int(diff)}" if diff > 0 else int(diff))
            delta_color = "normal" if boost_pct > 0 else "off"
            k4.metric("🚀 Performance Boost", f"{boost_pct:.1f}%", delta=f"{boost_pct:.1f}%",
                      delta_color=delta_color)

        st.divider()

        # --- Timeline Chart ---
        st.subheader(f"📈 The Road to Hosting: {full_country_name}")

        start_window = h_year - 24
        end_window = h_year + 12
        all_years_timeline = list(range(start_window, end_window + 4, 4))

        window_df = country_history[
            (country_history['year'] >= start_window) & (country_history['year'] <= end_window)].copy()
        window_df['prev_medals'] = window_df['medal'].shift(1)
        window_df['year_boost'] = (
                (window_df['medal'] - window_df['prev_medals']) / window_df['prev_medals'] * 100).fillna(0)

        window_df['tooltip_title'] = window_df['year'].apply(
            lambda y: f"HOST YEAR: {y}" if y == h_year else f"Year: {y}")

        fig_trend = px.line(window_df, x='year', y='medal', markers=True)

        fig_trend.update_traces(
            line_color='#1E90FF',
            marker_color='#1E90FF',
            marker_size=8,
            hovertemplate="<b>%{customdata[1]}</b><br>Medals: %{y}<br>Change: %{customdata[0]:.1f}%<extra></extra>",
            customdata=window_df[['year_boost', 'tooltip_title']]
        )

        max_medals = window_df['medal'].max() if not window_df.empty else 10
        if max_medals <= 15:
            y_dtick = 1
        elif max_medals <= 40:
            y_dtick = 5
        elif max_medals <= 100:
            y_dtick = 10
        else:
            y_dtick = 20

        tick_text = []
        for y in all_years_timeline:
            if y == h_year:
                tick_text.append(
                    f'<span style="color:#FF8C00; font-weight:bold; font-size:14px">{y}<br>HOST YEAR</span>')
            else:
                tick_text.append(str(y))

        fig_trend.update_layout(
            height=400,
            xaxis=dict(title="", tickmode='array', tickvals=all_years_timeline, ticktext=tick_text),
            yaxis=dict(title="Total Medals", dtick=y_dtick, rangemode="tozero"),
            plot_bgcolor='white',
            hovermode="closest"
        )
        fig_trend.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        st.plotly_chart(fig_trend, width='stretch')

        st.divider()

        # --- Sports Breakdown ---
        st.subheader(f"🏆 Where did {full_country_name} win the extra medals?")

        host_year_sports = medals_only[(medals_only['noc'] == h_noc) & (medals_only['year'] == h_year)]
        sport_counts = host_year_sports['sport'].value_counts().reset_index()
        sport_counts.columns = ['sport', 'count']

        prev_year = h_year - 4
        prev_year_sports = medals_only[(medals_only['noc'] == h_noc) & (medals_only['year'] == prev_year)]
        prev_counts = prev_year_sports['sport'].value_counts().reset_index()
        prev_counts.columns = ['sport', 'prev_count']

        sport_comp = pd.merge(sport_counts, prev_counts, on='sport', how='outer').fillna(0)
        sport_comp = sport_comp[sport_comp['count'] > sport_comp['prev_count']]
        sport_comp = sport_comp.sort_values('count', ascending=False).head(10)

        if not sport_comp.empty:
            fig_sports = go.Figure()
            fig_sports.add_trace(go.Bar(
                y=sport_comp['sport'], x=sport_comp['prev_count'],
                name=f"{prev_year}", orientation='h', marker_color='#9B59B6'
            ))
            fig_sports.add_trace(go.Bar(
                y=sport_comp['sport'], x=sport_comp['count'],
                name=f"{h_year} (Host)", orientation='h', marker_color='#00BFFF'
            ))

            fig_sports.update_layout(
                barmode='group', height=500, yaxis={'categoryorder': 'total ascending'},
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                plot_bgcolor='white'
            )
            fig_sports.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            st.plotly_chart(fig_sports, width='stretch')
        else:
            st.info(
                f"No specific sports found where {full_country_name} improved medal count compared to the previous Olympics.")