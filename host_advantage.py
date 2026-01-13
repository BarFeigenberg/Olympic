import plotly.graph_objects as go
from data_processor import *


def create_host_radar_chart(medals_only, host_data, h_noc, h_year, view_mode="All"):
    if medals_only.empty:
        return None

    country_years = sorted(medals_only[medals_only['noc'] == h_noc]['year'].unique())
    if not country_years:
        return None

    past_years = [y for y in country_years if y < h_year]
    future_years = [y for y in country_years if y > h_year]
    current_year = h_year if h_year in country_years else None
    categories = list(reversed(CATEGORY_ORDER))

    stats = {}
    for cat in categories:
        all_vals = []
        for y in country_years:
            val = get_medals_by_sport_category(medals_only, h_noc, y).get(cat, 0)
            all_vals.append(val)

        if not all_vals:
            stats[cat] = {'min': 0, 'max': 0, 'current': 0, 'past_avg': None, 'future_avg': None}
            continue

        stats[cat] = {
            'min': min(all_vals),
            'max': max(all_vals),
            'current': get_medals_by_sport_category(medals_only, h_noc, current_year).get(cat,
                                                                                          0) if current_year else 0,
            'past_avg': (sum(p_vals := [get_medals_by_sport_category(medals_only, h_noc, y).get(cat, 0) for y in
                                        past_years]) / len(p_vals)) if past_years else None,
            'future_avg': (sum(f_vals := [get_medals_by_sport_category(medals_only, h_noc, y).get(cat, 0) for y in
                                          future_years]) / len(f_vals)) if future_years else None
        }

    fig = go.Figure()

    # Colors Palette
    c_red = "#EE334E"
    c_blue = "#0081C8"
    c_green = "#00A651"
    c_range = "rgba(220, 220, 220, 0.6)"
    c_ticks = "rgba(150, 150, 150, 0.8)"

    for i, cat in enumerate(categories):
        s = stats[cat]
        if s['max'] > 0:
            # 1. Range Bar (The horizontal gray line)
            fig.add_trace(go.Scatter(
                x=[s['min'], s['max']], y=[i, i],
                mode='lines',
                line=dict(color=c_range, width=8),
                hoverinfo='skip', showlegend=False
            ))

            # 2. Range End Ticks (The "Dumbbell" brackets)
            fig.add_trace(go.Scatter(
                x=[s['min'], s['max']], y=[i, i],
                mode='markers',
                marker=dict(
                    symbol='line-ns-open',
                    size=14,
                    line_color=c_ticks,
                    color=c_ticks,
                    line_width=2
                ),
                showlegend=False, hoverinfo='skip'
            ))

        # 3. Past Average Marker (Red)
        if s['past_avg'] is not None:
            fig.add_trace(go.Scatter(
                x=[s['past_avg']], y=[i],
                mode='markers',
                marker=dict(symbol='line-ns', size=19, line_color=c_red, color=c_red, line_width=4),
                name='Past Avg', showlegend=True if i == 0 else False,
                hovertemplate=f"Past Avg: {s['past_avg']:.1f}<extra></extra>"
            ))

        # 4. Future Average Marker (Green)
        if s['future_avg'] is not None:
            fig.add_trace(go.Scatter(
                x=[s['future_avg']], y=[i],
                mode='markers',
                marker=dict(symbol='line-ns', size=19, line_color=c_green, color=c_green, line_width=4),
                name='Future Avg', showlegend=True if i == 0 else False,
                hovertemplate=f"Future Avg: {s['future_avg']:.1f}<extra></extra>"
            ))

        # 5. Host Year Marker (Blue)
        if current_year:
            fig.add_trace(go.Scatter(
                x=[s['current']], y=[i],
                mode='markers',
                marker=dict(symbol='line-ns', size=23, line_color=c_blue, color=c_blue, line_width=6),
                name=f'{h_year} (Host)', showlegend=True if i == 0 else False,
                hovertemplate=f"{h_year}: {s['current']}<extra></extra>"
            ))

    # Add legend entry for the gray range bar
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color='rgba(180, 180, 180, 0.8)', width=8),
        name='Historical Range',
        showlegend=True
    ))

    fig.update_layout(
        xaxis=dict(title="Number of Medals", showgrid=True, gridcolor='whitesmoke'),
        yaxis=dict(
            tickmode='array', tickvals=list(range(len(categories))), ticktext=categories,
            tickfont=dict(size=12, family="Arial Black")
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
        colorway=["#808080"],  # Prevents automatic color cycling
        height=600, margin=dict(l=20, r=20, t=100, b=20), plot_bgcolor='white'
    )

    return fig


def create_parallel_coordinates_chart(medals_only, host_data, selected_noc, focus_year=None):
    # --- FIX: Use medals_data (Tally) for accurate counts including Paris 2024 ---
    from data_processor import get_processed_medals_data
    medals_data = get_processed_medals_data()

    if medals_data.empty:
        return None

    # Calculate global totals per year from tally data
    global_counts = medals_data.groupby('year')['total'].sum().reset_index()
    global_counts.columns = ['year', 'global_total']

    # Get country-specific data
    country_data = medals_data[medals_data['country_noc'] == selected_noc]
    if country_data.empty:
        return None

    country_stats = country_data.groupby('year')['total'].sum().reset_index()
    country_stats.columns = ['year', 'country_total']

    df = pd.merge(country_stats, global_counts, on='year', how='left')
    df['percentage'] = (df['country_total'] / df['global_total']) * 100

    host_years = host_data[host_data['host_noc'] == selected_noc]['year'].unique()
    df['is_host'] = df['year'].apply(lambda x: 1 if x in host_years else 0)

    # Filter war years first AND 1906 (Intercalated)
    war_years = [1906, 1916, 1940, 1944]
    df = df[~df['year'].isin(war_years)].copy()

    # THE TRICK: Convert year to string to force categorical behavior
    df['year_str'] = df['year'].astype(str)
    
    # Update the timeline list to match strings (also filter 1906 from here)
    full_timeline_str = [str(y) for y in range(1896, 2025, 4) if y not in war_years and y != 1906]

    # Map years to a numeric sequence (0, 1, 2...)
    year_map = {year: i for i, year in enumerate(full_timeline_str)}
    df['year_index'] = df['year_str'].map(year_map)

    # Use GLOBAL range (1896-2024)
    global_min_year = 1896
    global_max_year = 2024
    global_range = global_max_year - global_min_year

    # Define helper to calculate NUMERICAL color value
    # Split Range Strategy (Normalized to 0-1 range):
    # 0.00 - 0.45: Standard Years (Red Saturation)
    # 0.50 - 0.90: Host Years (Green Saturation)
    # 1.00: Selected Focus Year (Blue)
    def get_color_val(row):
        # 0. Check for Focus Year (Absolute Top Priority)
        if focus_year and row['year'] == focus_year:
            return 1.0
            
        # Normalize year (0.0 to 1.0) based on global range
        norm = (row['year'] - global_min_year) / global_range
        norm = max(0.0, min(1.0, norm))
        
        if row['is_host']:
            # Map Host Years to 0.50 - 0.90 range
            return 0.50 + (norm * 0.40)
        else:
            # Map Standard Years to 0.00 - 0.45 range
            return norm * 0.45

    df['line_color_val'] = df.apply(get_color_val, axis=1)

    # Sort: Standard -> Host -> Focused Year (Last = Top)
    df = df.sort_values(by='line_color_val', ascending=True)

    # Custom Dual-Gradient Colorscale with BLUE/RED THEME + GREEN HIGHLIGHT
    # Strategy: "Standard=Blue(Lighter), Host=Red" + Opacity Gradients
    # MUST be strictly between 0 and 1
    custom_colorscale = [
        # --- STANDARD YEARS (0.0 - 0.45): LIGHT BLUE GRADIENT ---
        # Very Old (0.0): Very Pale Blue / SkyBlue
        [0.0, 'rgba(135, 206, 250, 0.3)'], 
        # Very New (0.45): Medium Blue (Medium Opacity)
        [0.45, 'rgba(0, 0, 180, 0.7)'],
        
        # --- HOST YEARS (0.50 - 0.90): RED GRADIENT ---
        # Very Old (0.50): Light Red (Visible)
        [0.50, 'rgba(255, 100, 100, 0.85)'],
        # Very New (0.90): Dark Red (Bold)
        [0.90, 'rgba(180, 0, 0, 1.0)'],
        
        # --- FOCUS YEAR (1.0): GREEN HIGHLIGHT ---
        # Hard start for Green segment at 0.95
        [0.95, 'rgba(0, 200, 0, 1.0)'],    # Bright Green (Lime/Green)
        [1.0, 'rgba(0, 200, 0, 1.0)']      # Bright Green
    ]

    fig = go.Figure(data=
    go.Parcoords(
        line=dict(
            color=df['line_color_val'],
            colorscale=custom_colorscale,
            cmin=0.0,
            cmax=1.0, # Strictly 0 to 1
            showscale=False
        ),
        dimensions=[
            dict(label='Year',
                 values=df['year_index'],
                 tickvals=list(year_map.values()),
                 ticktext=list(year_map.keys())),
            dict(range=[-1, 1.5],
                 tickvals=[-1, 0, 1, 1.5],
                 ticktext=['', 'No', 'Yes', ''],
                 label='Is Host?', values=df['is_host']),
            dict(label='Medals', values=df['country_total']),
            dict(label='Medals Share (%)', values=df['percentage'], tickformat='.1f')
        ],
        labelfont=dict(size=14, color='black', family="Arial Black"),
        tickfont=dict(size=10, color='black')
    )
    )

    fig.update_layout(
        height=800,
        margin=dict(l=80, r=80, t=80, b=40),
        paper_bgcolor="white",
        plot_bgcolor="white"
    )

    return fig


def create_timeline_selector(all_years, selected_year):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[str(y) for y in all_years], # Convert to string for categorical axis
        y=[0] * len(all_years),
        mode='lines+markers',
        marker=dict(
            size=14,
            color=['#D4AF37' if y == selected_year else '#E0E0E0' for y in all_years],
            line=dict(width=2, color='white')
        ),
        line=dict(color='#f0f0f0', width=2),
        hoverinfo='text',
        text=[f"Olympic Year: {y}" for y in all_years],
    ))

    fig.update_layout(
        height=120,
        margin=dict(l=40, r=40, t=20, b=40),
        xaxis=dict(
            type='category', # This removes the gaps between 1936 and 1948
            showgrid=False,
            showline=False,
            zeroline=False,
            tickangle=45,
            tickfont=dict(size=10, color='#666', family="Arial"),
            fixedrange=True
        ),
        yaxis=dict(showgrid=False, showline=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        paper_bgcolor='white',
        dragmode=False
    )

    return fig.update_layout(modebar_remove=['zoom', 'pan', 'select', 'lasso2d'])


def show_host_advantage(host_data, medals_only, country_ref):
    # --- 1. PREPARE DATA FOR DROPDOWN (Must be done before layout) ---
    noc_map = {}
    if not country_ref.empty:
        noc_map = dict(zip(country_ref['noc'], country_ref['country']))

    # Fix: Initialize h_year with a fallback value immediately
    h_year = 2024

    def get_label(row):
        full_name = noc_map.get(row['host_noc'], row['host_noc'])
        return f"{row['year']} - {row['host_city']} ({full_name})"

    host_data['label'] = host_data.apply(get_label, axis=1)
    options = sorted(host_data['label'].unique(), reverse=True)

    # --- 2. NEW LAYOUT: Title Left, Dropdown Right ---
    st.title("The Host Effect")
    st.markdown("Does hosting the Olympics actually guarantee more medals?")
    st.divider()

    # THE GLOBAL "BIG QUESTION" CHART ---
    st.subheader("The Big Picture: Does Hosting Pay Off?")

    # Calculate 'Lift %'
    host_data['lift_percent'] = (host_data['lift'] - 1) * 100

    host_data.loc[host_data['year'] == 1896, 'lift_percent'] = 0
    host_data['color'] = host_data['lift_percent'].apply(lambda x: '#00A651' if x >= 0 else '#EE334E')
    # Sort Data
    global_chart_data = host_data.sort_values('year').reset_index(drop=True)

    # --- TIMELINE SLIDER LOGIC ---

    # 1. Create Placeholder for Chart
    chart_place = st.empty()

    # --- 2. RENDER TIMELINE SELECTOR (Plotly Scatter Version) ---
    # 1. Create a complete timeline from 1896 to 2024 (every 4 years)
    war_years = [1906, 1916, 1940, 1944]
    full_timeline = [y for y in range(1896, 2025, 4) if y not in war_years and y != 1906]

    # 2. Render TIMELINE SELECTOR with the full timeline
    if 'last_center_year' not in st.session_state:
        st.session_state.last_center_year = 2024

    # Use full_timeline instead of all_years
    timeline_fig = create_timeline_selector(full_timeline, int(st.session_state.last_center_year))

    selected_point = st.plotly_chart(timeline_fig, width='stretch', on_select="rerun", key="timeline_chart")

    if selected_point and selected_point.get("selection") and selected_point["selection"]["points"]:
        new_year = int(selected_point["selection"]["points"][0]["x"])
        if new_year != st.session_state.last_center_year:
            st.session_state.last_center_year = new_year
            st.rerun()

    center_year = st.session_state.last_center_year
    # 3. Calculate Window for the Bar Chart
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

    # 4. Generate Enhanced Timeline Chart
    fig_global = go.Figure()

    filtered_global_data['host_country_name'] = filtered_global_data['host_noc'].map(noc_map).fillna(filtered_global_data['host_noc'])
    # Enhanced Bar chart with dynamic colors and borders
    fig_global.add_trace(go.Bar(
        x=filtered_global_data['year'],
        y=filtered_global_data['lift_percent'],
        marker=dict(
            color=filtered_global_data['color'],
            line=dict(color='white', width=1.5)  # Clean white border
        ),
        text=filtered_global_data['host_country_name'],
        textposition="outside",
        hovertemplate="<b>%{text} (%{x})</b><br>Performance Boost: %{y:.1f}%<extra></extra>"
    ))

    # Add a reference line for "No Change" (0%)
    fig_global.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)

    fig_global.update_layout(
        yaxis_title="Boost / Drop (%)",
        xaxis_title="",
        height=400,  # Made it slightly taller
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            type='category',
            categoryorder='array',
            categoryarray=[str(y) for y in full_timeline],
            fixedrange=True,
            showgrid=False,
            tickfont=dict(size=12, family="Arial Black")
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.2)',
            zeroline=False,
            ticksuffix="%"
        ),
        margin=dict(l=0, r=0, t=30, b=0)
    )

    # 5. Place Chart in placeholder (ABOVE slider)
    chart_place.plotly_chart(fig_global, width='stretch')
    st.caption("Tip: Select the year you want to see in the middle.")
    st.divider()

    col_sel, col_spacer = st.columns([1, 5])

    with col_sel:
        # Default to Sydney 2000 if available
        default_index = 0
        sydney_option = None
        
        for i, opt in enumerate(options):
            if "2000" in opt and "Sydney" in opt:
                default_index = i
                sydney_option = opt
                break
        
        # Initialize session state if not present
        if "host_event_selector" not in st.session_state:
             if sydney_option:
                 st.session_state.host_event_selector = sydney_option
             else:
                 st.session_state.host_event_selector = options[0] if options else None
        
        sel_event = st.selectbox("Select Host Event:", options, key="host_event_selector")

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

    st.write("")
    st.write("")
    # --- RADAR CHART: Medal Distribution by Sport Category ---
    if sel_event and h_noc and h_year:
        st.subheader(f"Medal Distribution by Sport Category: {full_country_name} - {h_year}")
        st.caption(f"Comparing {h_year} performance across sport categories")

        # --- 1. PRE-CALCULATE KPIS ---
        country_history = medals_only[medals_only['noc'] == h_noc].groupby('year')['medal'].count().reset_index()
        pre_years = country_history[(country_history['year'] < h_year) & (country_history['year'] >= h_year - 12)]
        avg_pre = pre_years['medal'].mean() if not pre_years.empty else 0
        diff = h_medals - avg_pre
        boost_pct = (diff / avg_pre * 100) if avg_pre > 0 else 0

        # --- 2. SHARED STYLING ---
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
        value_style = "font-size: 26px; font-weight: 600; color: rgb(49, 51, 63);"

        # Determine Boost Icon and Color
        boost_color = "#2ECC71" if boost_pct > 0 else "#E74C3C"
        boost_icon = "▲" if boost_pct > 0 else "▼"

        # --- 3. METRICS ROW ---
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.markdown(
                f'<div style="{card_style}"><div style="{label_style}">Host Year Medals </div><div style="{value_style}">{h_medals}</div></div>',
                unsafe_allow_html=True)
        with m2:
            st.markdown(
                f'<div style="{card_style}"><div style="{label_style}">Pre-Host Avg (12y) </div><div style="{value_style}">{avg_pre:.1f}</div></div>',
                unsafe_allow_html=True)
        with m3:
            st.markdown(
                f'<div style="{card_style}"><div style="{label_style}">Net Gain </div><div style="{value_style}">{" + " if diff > 0 else ""}{int(diff)}</div></div>',
                unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
                        <div style="{card_style}">
                            <div style="{label_style}">Performance Boost</div>
                            <div style="{value_style}">
                                {boost_pct:.1f}% 
                                <span style="color: {boost_color}; font-size: 18px; margin-left: 5px;">{boost_icon}</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

        st.write("")  # Spacer

        # --- 4. FULL WIDTH CHART ---
        # Note: view_mode is set to "All" and the radio buttons were removed
        radar_fig = create_host_radar_chart(medals_only, host_data, h_noc, h_year, view_mode="All")
        if radar_fig:
            st.plotly_chart(radar_fig, width='stretch')

    st.divider()
    if sel_event:
        st.subheader(f"Olympic Journey Timeline - {full_country_name}")

        col_chart, col_info = st.columns([5, 1], gap="small")
        country_years = sorted([y for y in medals_only[medals_only['noc'] == h_noc]['year'].unique() if y != 1906])

        with col_info:
            focus_year = st.selectbox("Year:", ["All"] + list(reversed(country_years)))
            focus_year_val = None if focus_year == "All" else int(focus_year)

            if focus_year_val:
                # Calculate data using Tally (includes Paris 2024)
                from data_processor import get_processed_medals_data
                medals_data = get_processed_medals_data()

                # Country medals for the focused year
                y_data = medals_data[
                    (medals_data['country_noc'] == h_noc) & (medals_data['year'] == focus_year_val)]
                m_count = int(y_data['total'].sum()) if not y_data.empty else 0

                # Global total for the focused year
                g_total = medals_data[medals_data['year'] == focus_year_val]['total'].sum()
                share = (m_count / g_total * 100) if g_total > 0 else 0
                is_host_year = focus_year_val in host_data[host_data['host_noc'] == h_noc]['year'].values

                # Smaller Card Styling
                side_card = """
                    background-color: #ffffff;
                    border: 1px solid #f0f2f6;
                    padding: 10px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
                    margin-bottom: 8px;
                    text-align: center;
                """
                side_label = "font-size: 15px; color: #666; margin-bottom: 2px;"
                side_value = "font-size: 21px; font-weight: bold; color: #333;"

                # Render Cards
                st.markdown(
                    f'<div style="{side_card}"><div style="{side_label}">Medals</div><div style="{side_value}">{m_count}</div></div>',
                    unsafe_allow_html=True)
                st.markdown(
                    f'<div style="{side_card}"><div style="{side_label}">Share</div><div style="{side_value}">{share:.1f}%</div></div>',
                    unsafe_allow_html=True)

                host_color = "#00A651" if is_host_year else "#EE334E"
                host_text = "Yes" if is_host_year else "No"
                st.markdown(
                    f'<div style="{side_card}"><div style="{side_label}">Host</div><div style="{side_value}; color: {host_color};">{host_text}</div></div>',
                    unsafe_allow_html=True)
            else:
                st.caption("Select a year to focus on.")

            focus_year_val = None if focus_year == "All" else int(focus_year)
            st.write("")

        with col_chart:
            fig_parcoords = create_parallel_coordinates_chart(medals_only, host_data, h_noc, focus_year_val)
            if fig_parcoords:
                st.plotly_chart(fig_parcoords, width='stretch')
