import plotly.graph_objects as go
from data_processor import *


def create_host_radar_chart(medals_only, host_data, h_noc, h_year, view_mode="All"):
    """
    Create a Horizontal Range Chart (Dumbbell Plot) showing medal distribution.
    Ranges show the country's historical Min-Max medals per category.
    Markers show Current (Host), Past Avg, and Future Avg.
    """
    if medals_only.empty:
        return None
    
    # Get all years for this country
    country_years = sorted(medals_only[medals_only['noc'] == h_noc]['year'].unique())
    
    if not country_years:
        return None
    
    # Define groups
    past_years = [y for y in country_years if y < h_year]
    future_years = [y for y in country_years if y > h_year]
    current_year = h_year if h_year in country_years else None
    
    categories = list(reversed(CATEGORY_ORDER)) # Reverse for Top-to-Bottom plotting
    
    # --- Calculate Statistics per Category ---
    stats = {}
    
    for cat in categories:
        # Get medals for ALL years to find Min/Max Range
        all_vals = []
        for y in country_years:
            val = get_medals_by_sport_category(medals_only, h_noc, y).get(cat, 0)
            all_vals.append(val)
        
        if not all_vals:
             stats[cat] = {'min': 0, 'max': 0, 'current': 0, 'past_avg': None, 'future_avg': None}
             continue

        # range
        min_val = min(all_vals)
        max_val = max(all_vals)
        
        # Current Value
        curr_val = 0
        if current_year:
             curr_val = get_medals_by_sport_category(medals_only, h_noc, current_year).get(cat, 0)
        
        # Past Avg
        p_avg = None
        if past_years:
            p_vals = []
            for y in past_years:
                p_vals.append(get_medals_by_sport_category(medals_only, h_noc, y).get(cat, 0))
            p_avg = sum(p_vals) / len(p_vals) if p_vals else 0

        # Future Avg
        f_avg = None
        if future_years:
            f_vals = []
            for y in future_years:
                f_vals.append(get_medals_by_sport_category(medals_only, h_noc, y).get(cat, 0))
            f_avg = sum(f_vals) / len(f_vals) if f_vals else 0
            
        stats[cat] = {
            'min': min_val, 'max': max_val,
            'current': curr_val,
            'past_avg': p_avg,
            'future_avg': f_avg
        }
    first_future_year = True
    first_future_host_year = True
    
    # Map categories to numeric indices
    cat_y_map = {cat: i for i, cat in enumerate(categories)}
    
    fig = go.Figure()
    
    # --- Calculate Dynamic Y-Offsets for Collision Avoidance ---
    # We want to keep points on the line (offset 0) unless they collide (close in X).
    # If they collide, we spread them vertically.
    
    y_offsets_data = {cat: {'past': 0, 'future': 0, 'current': 0} for cat in categories}
    
    for cat in categories:
        points = []
        if view_mode in ["Past + Current", "All"] and stats[cat]['past_avg'] is not None:
            points.append({'type': 'past', 'val': stats[cat]['past_avg']})
        if view_mode in ["Current + Future", "All"] and stats[cat]['future_avg'] is not None:
             points.append({'type': 'future', 'val': stats[cat]['future_avg']})
        if current_year and view_mode in ["Past + Current", "Current + Future", "All"]:
             points.append({'type': 'current', 'val': stats[cat]['current']})
        
        if not points:
            continue
            
        # Sort by value to find clusters
        points.sort(key=lambda x: x['val'])
        
        # Simple clustering: if dist < threshold, group them
        threshold = 2.0 # Medal count threshold for overlap
        clusters = []
        if points:
            curr_cluster = [points[0]]
            for i in range(1, len(points)):
                if (points[i]['val'] - points[i-1]['val']) < threshold:
                    curr_cluster.append(points[i])
                else:
                    clusters.append(curr_cluster)
                    curr_cluster = [points[i]]
            clusters.append(curr_cluster)
        
        # Assign offsets within clusters
        for cluster in clusters:
            count = len(cluster)
            if count == 1:
                y_offsets_data[cat][cluster[0]['type']] = 0
            elif count == 2:
                # Spread: -0.15, +0.15
                offsets = [-0.15, 0.15]
                # Sort cluster by type priority for consistent stacking? Or random?
                # Let's keep input order or val order. 
                # cluster is sorted by val.
                for idx, pt in enumerate(cluster):
                    y_offsets_data[cat][pt['type']] = offsets[idx]
            elif count == 3:
                # Spread: -0.2, 0, +0.2
                offsets = [-0.25, 0, 0.25]
                for idx, pt in enumerate(cluster):
                    y_offsets_data[cat][pt['type']] = offsets[idx]

    # 1. Draw Range Lines (Background) at Center (y=i)
    for i, cat in enumerate(categories):
        s = stats[cat]
        if s['max'] > 0:
            # Add line shape manually
            fig.add_shape(
                type="line",
                x0=s['min'], y0=i, x1=s['max'], y1=i,
                line=dict(color="rgba(220, 220, 220, 0.6)", width=8),
                layer="below"
            )
            
            # Add vertical ticks at ends
            fig.add_trace(go.Scatter(
                x=[s['min'], s['max']], y=[i, i],
                mode='markers',
                marker=dict(symbol='line-ns-open', size=12, color='silver', line_width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

    # Add hidden trace for legend entry 'Historical Range'
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color="rgba(220, 220, 220, 0.8)", width=8),
        name='Historical Range'
    ))

    # 2. Add Markers with Dynamic Offsets
    
    # Past Avg (Red)
    x_past = []
    y_past = []
    t_past = []
    if view_mode in ["Past + Current", "All"]:
        for cat in categories:
            if stats[cat]['past_avg'] is not None:
                x_past.append(stats[cat]['past_avg'])
                # Use dynamic offset
                y_past.append(cat_y_map[cat] + y_offsets_data[cat]['past'])
                t_past.append(f"{cat}<br>Past Avg: {stats[cat]['past_avg']:.1f}")

        fig.add_trace(go.Scatter(
            x=x_past, y=y_past,
            mode='markers',
            name='Past Avg',
            marker=dict(symbol='circle', color='#B22222', size=11, line=dict(color='black', width=1)), # Red Circle
            hovertemplate="%{text}<extra></extra>",
            text=t_past
        ))

    # Future Avg (Green)
    x_future = []
    y_future = []
    t_future = []
    if view_mode in ["Current + Future", "All"]:
        for cat in categories:
            if stats[cat]['future_avg'] is not None:
                x_future.append(stats[cat]['future_avg'])
                y_future.append(cat_y_map[cat] + y_offsets_data[cat]['future'])
                t_future.append(f"{cat}<br>Future Avg: {stats[cat]['future_avg']:.1f}")

        fig.add_trace(go.Scatter(
            x=x_future, y=y_future,
            mode='markers',
            name='Future Avg',
            marker=dict(symbol='circle', color='#00C853', size=11, line=dict(color='black', width=1)), # Green Circle
            hovertemplate="%{text}<extra></extra>",
            text=t_future
        ))

    # Current (Gold)
    x_curr = []
    y_curr = []
    t_curr = []
    if current_year and view_mode in ["Past + Current", "Current + Future", "All"]:
        for cat in categories:
            val = stats[cat]['current']
            x_curr.append(val)
            y_curr.append(cat_y_map[cat] + y_offsets_data[cat]['current'])
            t_curr.append(f"{cat}<br>{current_year}: {val} (Host)")

        fig.add_trace(go.Scatter(
            x=x_curr, y=y_curr,
            mode='markers',
            name=f'{current_year} (Host)',
            marker=dict(symbol='circle', color='#FFD700', size=15, line=dict(color='black', width=1)), # Gold Circle
            hovertemplate="%{text}<extra></extra>",
            text=t_curr
        ))

    fig.update_layout(
        title=dict(text="Medal Range per Sport Category", font=dict(size=18)),
        xaxis=dict(
            title="Number of Medals",
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.2)'
        ),
        yaxis=dict(
            title="",
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            tickmode='array',
            tickvals=list(range(len(categories))),
            ticktext=categories,
            tickfont=dict(color='black', size=13, family='Arial Black, sans-serif')
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=600,
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor='white'
    )
    
    return fig




def selection_dropdown(host_data, country_ref):
    # --- CSS FOR SELECTBOX CARD ---
    st.markdown("""
        <style>
            /* Target the Selectbox container */
            [data-testid="stSelectbox"] {
                background-color: white;
                padding: 10px;           /* Inner spacing */
                border-radius: 10px;     /* Rounded corners */
                border: 1px solid #dedede; /* Thin grey border */
                box-shadow: 2px 2px 5px rgba(0,0,0,0.1); /* Shadow */
            }

            /* Force the label text (title) to be dark grey */
            [data-testid="stSelectbox"] label {
                color: #444 !important;
            }
        </style>
        """, unsafe_allow_html=True)


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

    # --- RADAR CHART: Medal Distribution by Sport Category ---
    if sel_event and h_noc and h_year:
        st.subheader(f"🎯 Medal Distribution by Sport Category: {full_country_name} - {h_year}")

        # --- PRE-CALCULATE KPIS FOR METRICS ---
        country_history = medals_only[medals_only['noc'] == h_noc].groupby('year')['medal'].count().reset_index()
        pre_years = country_history[(country_history['year'] < h_year) & (country_history['year'] >= h_year - 12)]
        avg_pre = pre_years['medal'].mean() if not pre_years.empty else 0
        diff = h_medals - avg_pre
        boost_pct = (diff / avg_pre * 100) if avg_pre > 0 else 0

        # --- CSS STYLING FOR METRICS (Card Style) ---
        st.markdown("""
                <style>
                    /* Style the metric container */
                    [data-testid="stMetric"] {
                        background-color: white;
                        border: 1px solid #dedede;
                        padding: 10px;
                        border-radius: 10px;
                        color: black;
                        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
                        margin-bottom: 10px; /* Space between vertical cards */
                    }
                    /* Force label color */
                    [data-testid="stMetricLabel"] p {
                        color: #444 !important;
                    }
                    /* Force value color */
                    [data-testid="stMetricValue"] div {
                        color: black !important;
                    }

                    /* RADIO BUTTON SMALLER STYLE */
                    div[role="radiogroup"] label p {
                        font-size: 13px !important;
                    }
                    div[role="radiogroup"] {
                        gap: 0px !important;
                    }
                </style>
                """, unsafe_allow_html=True)
        
        # --- LAYOUT: [Chart 2/3] | [Metrics 1/3] ---
        col_chart, col_metrics = st.columns([2, 1], gap="medium")

        with col_chart:
            # Top controls - just view mode
            ctrl_col1, ctrl_col2 = st.columns([2, 1])
            with ctrl_col1:
                st.caption(f"Comparing {h_year} performance across sport categories")
            with ctrl_col2:
                view_mode = st.radio(
                    "View Mode:",
                    ["All", "Past + Current", "Current + Future"],
                    horizontal=False,
                    label_visibility="collapsed" # Save space
                )
            
            # Create and display radar chart
            radar_fig = create_host_radar_chart(medals_only, host_data, h_noc, h_year, view_mode)
            if radar_fig:
                st.plotly_chart(radar_fig, use_container_width=True)
            else:
                st.info("No medal data available for this selection.")

        with col_metrics:
            st.write("#### Performance Stats")
            st.metric("🏅 Host Year Medals", h_medals)
            st.metric("📊 Pre-Host Avg (12y)", f"{avg_pre:.1f}")
            st.metric("📈 Net Gain", f"+{int(diff)}" if diff > 0 else int(diff))
            
            delta_color = "normal" if boost_pct > 0 else "off"
            st.metric("🚀 Performance Boost", f"{boost_pct:.1f}%", delta=f"{boost_pct:.1f}%",
                      delta_color=delta_color)
        
        st.divider()

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
        
        # (KPI Logic moved up)
        
        st.divider()

        # --- Timeline Chart ---
        st.subheader(f"📈 The Road to Hosting: {full_country_name}")

        start_window = h_year - 24
        end_window = h_year + 12
        all_years_timeline = list(range(start_window, end_window + 4, 4))

        country_history = medals_only[medals_only['noc'] == h_noc].groupby('year')['medal'].count().reset_index()
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