import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from data_processor import get_medals_by_sport_category, CATEGORY_ORDER


def create_host_radar_chart(medals_only, host_data, h_noc, h_year, view_mode="All"):
    """
    Create a radar chart showing medal distribution by sport category.
    Normalizes each category to its own max for better visualization.
    """
    if medals_only.empty:
        return None
    
    # Get all years for this country
    country_years = sorted(medals_only[medals_only['noc'] == h_noc]['year'].unique())
    
    if not country_years:
        return None
    
    # Split into past, current, future
    past_years = [y for y in country_years if y < h_year]
    future_years = [y for y in country_years if y > h_year]
    current_year = h_year if h_year in country_years else None
    
    # Get host years for this country
    host_years_for_country = set()
    if not host_data.empty and h_noc:
        host_years_for_country = set(host_data[host_data['host_noc'] == h_noc]['year'].tolist())
    
    categories = CATEGORY_ORDER
    
    # Calculate MAX per category across all years (for normalization)
    max_per_category = {cat: 1 for cat in categories}  # Start with 1 to avoid division by 0
    for year in country_years:
        cat_medals = get_medals_by_sport_category(medals_only, h_noc, year)
        for cat in categories:
            max_per_category[cat] = max(max_per_category[cat], cat_medals.get(cat, 0) + 1)
    
    def normalize_values(raw_values):
        """Normalize values to 0-100 scale based on each category's max"""
        return [100 * raw_values.get(cat, 0) / max_per_category[cat] for cat in categories]
    
    fig = go.Figure()
    
    # --- PAST EVENTS (Burgundy thin lines, Gold for host years) ---
    # Different BRONZE/COPPER shades for other host events (to distinguish from current GOLD)
    bronze_shades = ['rgba(205, 127, 50, 0.8)',   # Bronze
                     'rgba(184, 115, 51, 0.8)',   # Copper
                     'rgba(210, 105, 30, 0.8)',   # Chocolate
                     'rgba(184, 134, 11, 0.8)']   # Dark Goldenrod
    host_count = 0
    
    # Track if we've added the first past year (for legend)
    first_past_year = True
    first_host_year = True
    
    if view_mode in ["Past + Current", "All"] and past_years:
        for year in past_years:
            is_host_year = year in host_years_for_country
                
            cat_medals = get_medals_by_sport_category(medals_only, h_noc, year)
            values = normalize_values(cat_medals)
            values.append(values[0])  # Close the radar
            
            if is_host_year:
                line_color = bronze_shades[host_count % len(bronze_shades)]
                line_width = 3  # Thicker for host
                host_count += 1
            else:
                line_color = 'rgba(139, 0, 0, 0.2)'  # Burgundy, more transparent
                line_width = 1  # Thinnest possible
            
            # Build custom hover data
            raw = [cat_medals.get(cat, 0) for cat in categories]
            
            # Determine legend group and name for Host vs Past
            if is_host_year:
                trace_name = 'Other Host Years' if first_host_year else f'{year} (Host)'
                grp = 'host_years'
                show_leg = first_host_year  # Only show specific legend item once
                first_host_year = False
                vis = True
            else:
                trace_name = 'Past Years' if first_past_year else f'{year}'
                grp = 'past_years'
                show_leg = first_past_year
                first_past_year = False
                vis = True

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                mode='lines',
                name=trace_name,
                legendgroup=grp,
                line=dict(color=line_color, width=line_width),
                customdata=raw + [raw[0]],
                hovertemplate=f"<b>{year}</b><br>%{{theta}}: %{{customdata}} medals<extra></extra>",
                showlegend=show_leg,
                visible=vis
            ))
        
        # Past Average (Dark Red line)
        if len(past_years) >= 1:
            avg_raw = {cat: 0 for cat in categories}
            for year in past_years:
                cat_medals = get_medals_by_sport_category(medals_only, h_noc, year)
                for cat in categories:
                    avg_raw[cat] += cat_medals.get(cat, 0)
            avg_raw = {cat: v / len(past_years) for cat, v in avg_raw.items()}
            
            values = normalize_values(avg_raw)
            values.append(values[0])
            raw = [avg_raw[cat] for cat in categories]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                mode='lines',
                name='Past Average',
                line=dict(color='#B22222', width=5),  # FireBrick red, thick
                customdata=raw + [raw[0]],
                hovertemplate="<b>Past Average</b><br>%{theta}: %{customdata:.1f} medals<extra></extra>",
                showlegend=True
            ))
    
    # --- FUTURE EVENTS (Green thin lines) ---
    first_future_year = True
    if view_mode in ["Current + Future", "All"] and future_years:
        for year in future_years:
            cat_medals = get_medals_by_sport_category(medals_only, h_noc, year)
            values = normalize_values(cat_medals)
            values.append(values[0])
            raw = [cat_medals.get(cat, 0) for cat in categories]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                mode='lines',
                name='Future Years' if first_future_year else f'{year}',
                legendgroup='future_years',
                line=dict(color='rgba(50, 205, 50, 0.3)', width=1),  # More transparent, thinnest
                customdata=raw + [raw[0]],
                hovertemplate=f"<b>{year}</b><br>%{{theta}}: %{{customdata}} medals<extra></extra>",
                showlegend=first_future_year,
                visible=True
            ))
            first_future_year = False
        
        # Future Average (Green line)
        if len(future_years) >= 1:
            avg_raw = {cat: 0 for cat in categories}
            for year in future_years:
                cat_medals = get_medals_by_sport_category(medals_only, h_noc, year)
                for cat in categories:
                    avg_raw[cat] += cat_medals.get(cat, 0)
            avg_raw = {cat: v / len(future_years) for cat, v in avg_raw.items()}
            
            values = normalize_values(avg_raw)
            values.append(values[0])
            raw = [avg_raw[cat] for cat in categories]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                mode='lines',
                name='Future Average',
                line=dict(color='#00C853', width=5),  # Emerald green, thick
                customdata=raw + [raw[0]],
                hovertemplate="<b>Future Average</b><br>%{theta}: %{customdata:.1f} medals<extra></extra>",
                showlegend=True
            ))
    
    # --- CURRENT EVENT (Gold, thickest line) ---
    if current_year and view_mode in ["Past + Current", "Current + Future", "All"]:
        cat_medals = get_medals_by_sport_category(medals_only, h_noc, current_year)
        values = normalize_values(cat_medals)
        values.append(values[0])
        raw = [cat_medals.get(cat, 0) for cat in categories]
        
        is_host = current_year in host_years_for_country
        name = f'{current_year} (Host)' if is_host else f'{current_year}'
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            mode='lines+markers',
            name=name,
            line=dict(color='#FFD700', width=7),  # THICKEST (Increased to 7)
            marker=dict(size=10, color='#FFD700'),
            customdata=raw + [raw[0]],
            hovertemplate=f"<b>{name}</b><br>%{{theta}}: %{{customdata}} medals<extra></extra>",
            showlegend=True
        ))
    
    # Layout - range is now 0-100 (normalized)
    fig.update_layout(
        polar=dict(
            gridshape='linear',  # Polygon shape instead of circle!
            radialaxis=dict(
                visible=True,
                showline=False,
                gridcolor='rgba(0,0,0,0.15)',
                range=[0, 105],
                tickvals=[25, 50, 75, 100],
                ticktext=['25%', '50%', '75%', 'Max'],
                tickfont=dict(size=10, color="gray")
            ),
            angularaxis=dict(
                gridcolor='rgba(0,0,0,0.3)',
                tickfont=dict(size=14, color="black")
            ),
            hole=0.02
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.08,
            xanchor="center",
            x=0.5,
            itemclick='toggle',
            itemdoubleclick='toggleothers',
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1,
            font=dict(size=11),
            itemsizing='constant',
            tracegroupgap=5,
            title=dict(text='💡 Click to toggle', font=dict(size=10))
        ),
        height=600,
        margin=dict(t=20, b=20, l=35, r=35)  # Tight margins for bigger chart
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
                    ["Past + Current", "Current + Future", "All"],
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