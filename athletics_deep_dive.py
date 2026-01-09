import os
import base64
import numpy as np
import plotly.graph_objects as go
from data_loader import *


def get_local_athlete_image_html(athlete_name):
    """
    Reads a local image and converts it to a base64 string
    to display via HTML, bypassing Python's image library limits (AVIF support).
    """
    filename_base = athlete_name.lower().replace(' ', '-')
    extensions = ['.avif', '.jpg', '.jpeg', '.png']
    base_path = "assets/athletes"

    for ext in extensions:
        full_path = os.path.join(base_path, f"{filename_base}{ext}")
        if os.path.exists(full_path):
            with open(full_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode()

            mime_type = "image/avif" if ext == ".avif" else f"image/{ext[1:]}"
            # Added max-height and object-fit to ensure the profile column stays aligned
            return f'<img src="data:{mime_type};base64,{encoded_string}"style="width:100%; max-height:380px; object-fit: cover; border-radius: 10px; margin-bottom: 10px;">'
    return None


def show_athletics_deep_dive(athletics_df):
    bio_df = load_athlete_bio_data()
    athletics_df = athletics_df.rename(columns={'numericresult': 'numeric_result'})
    if athletics_df is None or athletics_df.empty:
        st.error("⚠️ Data not found! Please check 'results.csv'.")
        return

    # Data Cleaning
    athletics_df['gender'] = athletics_df['gender'].astype(str).str.strip().str.upper()
    events = sorted(athletics_df['baseevent'].unique().tolist())

    # --- 1. SELECTION FILTERS (UPDATED LAYOUT) ---
    col_title, col_spacer, col_event = st.columns([5, 2, 2], gap="small")

    with col_title:
        st.title("🏃 Athletics Analysis")

    with col_spacer:
        st.empty()

    with col_event:
        st.write("")
        st.write("")
        default_idx = events.index('100M') if '100M' in events else 0
        e_name = st.selectbox("Select Event:", events, index=default_idx)

        # Filter by Event ONLY (Keep both genders)
        vdf = athletics_df[athletics_df['baseevent'] == e_name].copy()

        # Create a nice label for the legend
        vdf['gender_label'] = vdf['gender'].map({
            'M': 'Men', 'MEN': 'Men', 'MALE': 'Men',
            'W': 'Women', 'WOMEN': 'Women', 'FEMALE': 'Women'
        }).fillna(vdf['gender'])

    vdf = vdf.dropna(subset=['numeric_result'])

    if vdf.empty:
        st.warning(f"No results found for {e_name}.")
    else:
        # --- 2. RECORD CALCULATIONS ---
        is_high = any(x in e_name for x in ['Throw', 'Jump', 'Vault', 'athlon'])
        unit_title = "Points" if "athlon" in e_name else ("Meters" if is_high else "Seconds")
        if not is_high and vdf['numeric_result'].max() > 100:
            unit_title = "Time"

        # Record progression logic (Cumulative Min/Max)
        if is_high:
            yearly_best = vdf.groupby(['year', 'gender_label'])['numeric_result'].max().reset_index()
            yearly_best['running_record'] = yearly_best.groupby('gender_label')['numeric_result'].cummax()
        else:
            yearly_best = vdf.groupby(['year', 'gender_label'])['numeric_result'].min().reset_index()
            yearly_best['running_record'] = yearly_best.groupby('gender_label')['numeric_result'].cummin()

        yearly_best = yearly_best.sort_values('year')
        record_breaks = yearly_best[yearly_best['numeric_result'] == yearly_best['running_record']].copy()
        record_breaks = record_breaks.drop_duplicates(subset=['gender_label', 'numeric_result'], keep='first')
        record_breaks = record_breaks.merge(vdf[['year', 'gender_label', 'numeric_result', 'name']],
                                            on=['year', 'gender_label', 'numeric_result'], how='left')

        current_record_holder = record_breaks.sort_values('year').groupby('gender_label').tail(1)
        historical_record_breaks = record_breaks.drop(current_record_holder.index)

        # --- 3. UPPER SECTION: TREND CHART & PROFILE ---
        col_graph, col_info = st.columns([3, 1])

        with col_graph:
            st.subheader(f"Record Progression: {e_name}")

            # Define Colors: Pink for Women, Blue for Men
            color_map = {'Men': '#1E90FF', 'Women': '#FF69B4'}

            # Main Line Chart
            fig_trend = px.line(
                yearly_best,
                x='year',
                y='numeric_result',
                color='gender_label',  # This creates the Legend automatically
                color_discrete_map=color_map,
                labels={'gender_label': 'Gender'}
            )

            # Style the line and markers
            fig_trend.update_traces(line_width=2, marker_size=8)

            # Add Historical Record Markers (DodgerBlue dots)
            fig_trend.add_scatter(
                x=historical_record_breaks['year'], y=historical_record_breaks['numeric_result'],
                mode='markers',
                marker=dict(size=8, color='#1E90FF', line=dict(width=1, color='white')),
                customdata=historical_record_breaks['name'],
                name='Past Record', showlegend=False
            )

            # Add Current Record Highlight (DarkOrange #FF8C00 - same as "Host Year" in reference)
            fig_trend.add_scatter(
                x=current_record_holder['year'], y=current_record_holder['numeric_result'],
                mode='markers',
                marker=dict(size=14, color='#FF8C00', line=dict(width=2, color='white')),
                customdata=current_record_holder['name'],
                name='Olympic Record'
            )

            # Create the X-Axis timeline (All Olympic Years)
            min_year = int(vdf['year'].min())
            max_year = 2024
            timeline_years = list(range(1896, max_year + 4, 4))
            timeline_years = [y for y in timeline_years if y >= min_year - 4]

            # Highlight Current Record Year in Orange on the X-axis
            current_rec_years = current_record_holder['year'].tolist()
            tick_text = []
            for y in timeline_years:
                if y in current_rec_years:
                    tick_text.append(
                        f'<span style="color:#FF8C00; font-weight:bold; font-size:12px">{y}<br>RECORD</span>')
                else:
                    tick_text.append(str(y))

            # Reference Chart Styling: White background and LightGray grids
            fig_trend.update_layout(
                height=500,  # Fixed height to align with profile column
                plot_bgcolor='white',
                xaxis=dict(
                    tickmode='array', tickvals=timeline_years, ticktext=tick_text,
                    gridcolor='white', showline=True, linecolor='lightgray'
                ),
                yaxis=dict(
                    title=unit_title, showgrid=True, gridwidth=1, gridcolor='lightgray',
                    rangemode="tozero" if is_high else "normal"
                ),
                hovermode="closest",
                clickmode='event+select'
            )

            if not is_high:
                fig_trend.update_yaxes(autorange="reversed")

            selected_point = st.plotly_chart(fig_trend, width='stretch', on_select="rerun",
                                             key="athlete_selector")

        with col_info:
            st.subheader("👤 Athlete Profile")
            if selected_point and "selection" in selected_point and selected_point["selection"]["points"]:
                point_data = selected_point["selection"]["points"][0]
                athlete_name = point_data.get("customdata")

                if athlete_name:
                    st.write(f"**Name:** {athlete_name}")
                    st.write(f"**Year:** {point_data['x']}")
                    st.write(f"**Result:** {point_data['y']} {unit_title}")

                    # Base64 Image Display (Bypasses AVIF identify errors)
                    img_html = get_local_athlete_image_html(athlete_name)
                    if img_html:
                        st.markdown(img_html, unsafe_allow_html=True)
                    else:
                        st.image("https://cdn-icons-png.flaticon.com/512/847/847969.png", width=120)
                        st.info("Photo not found in assets.")
                else:
                    st.info("Click a record point (dot) to see details.")
            else:
                st.info("Click a marker on the graph to view the athlete's details.")

        # --- 4. LOWER SECTION: PHYSICAL ANALYSIS (GENDER SPECIFIC) ---
        st.divider()
        st.subheader("Physical Analysis")
        st.caption("Analyze the ideal body type for this event by gender.")

        if not bio_df.empty:
            vdf_phys = vdf[vdf['year'] >= 1960].copy()

            vdf_phys['name_clean'] = vdf_phys['name'].astype(str).str.lower().str.strip()
            bio_df['name_clean'] = bio_df['name'].astype(str).str.lower().str.strip()

            bio_subset = bio_df[['name_clean', 'born', 'height', 'weight']].copy()
            bio_subset['weight'] = pd.to_numeric(bio_subset['weight'], errors='coerce')
            bio_subset['height'] = pd.to_numeric(bio_subset['height'], errors='coerce')

            merged = pd.merge(vdf_phys, bio_subset, on='name_clean', how='inner')
            merged['born'] = pd.to_datetime(merged['born'], errors='coerce')
            merged['Age'] = merged['year'] - merged['born'].dt.year

            merged = merged.dropna(subset=['height', 'weight', 'Age', 'numeric_result'])

            # --- GENDER SELECTION ---
            gender_options = sorted(merged['gender_label'].unique())
            selected_gender = st.radio("Select Gender for Analysis:", gender_options, horizontal=True)

            merged_filtered = merged[merged['gender_label'] == selected_gender].copy()

            if not merged_filtered.empty:
                # --- 1. MEDAL & COUNTRY SETUP ---
                def format_medal(m):
                    m_str = str(m).lower()
                    if 'gold' in m_str or m_str == 'g': return '🥇 Gold'
                    if 'silver' in m_str or m_str == 's': return '🥈 Silver'
                    if 'bronze' in m_str or m_str == 'b': return '🥉 Bronze'
                    return 'No Medal'

                merged_filtered['Medal Display'] = merged_filtered['medal'].apply(format_medal)

                if 'team' in merged_filtered.columns:
                    merged_filtered['country'] = merged_filtered['team']
                else:
                    merged_filtered['country'] = merged_filtered['nationality']

                # --- 2. COLOR PALETTES ---
                blue_shades = [
                    '#deebf7', '#c6dbef', '#9ecae1', '#6baed6',
                    '#4292c6', '#2171b5', '#08519c', '#08306b'
                ]
                pink_shades = [
                    '#ffe8eb', '#fde0dd', '#fcc5c0', '#fa9fb5',
                    '#f768a1', '#dd3497', '#ae017e', '#7a0177'
                ]

                if 'Women' in str(selected_gender) or 'Female' in str(selected_gender):
                    current_shades = pink_shades
                else:
                    current_shades = blue_shades

                # --- 3. CREATE READABLE RANGES ---
                try:
                    merged_filtered['tier_interval'] = pd.qcut(merged_filtered['numeric_result'], q=8,
                                                               duplicates='drop')
                except ValueError:
                    merged_filtered['tier_interval'] = pd.qcut(merged_filtered['numeric_result'], q=4,
                                                               duplicates='drop')

                unique_intervals = sorted(merged_filtered['tier_interval'].unique())
                num_tiers = len(unique_intervals)

                label_map = {}
                color_map = {}
                ordered_labels = []

                for i, interval in enumerate(unique_intervals):
                    label_text = f"{interval.left:.2f} - {interval.right:.2f}"

                    if is_high:
                        color_idx = int((i / (num_tiers - 1)) * 7) if num_tiers > 1 else 7
                        ordered_labels.insert(0, label_text)
                    else:
                        color_idx = int(((num_tiers - 1 - i) / (num_tiers - 1)) * 7) if num_tiers > 1 else 7
                        ordered_labels.append(label_text)

                    final_color = current_shades[color_idx]
                    label_map[interval] = label_text
                    color_map[label_text] = final_color

                merged_filtered['Result Range'] = merged_filtered['tier_interval'].map(label_map)

                # DROP INTERVAL COLUMN
                merged_filtered = merged_filtered.drop(columns=['tier_interval'])

                # --- 4. CREATE CHART ---
                fig_phys = px.scatter(
                    merged_filtered,
                    x="weight",
                    y="height",
                    color="Result Range",
                    color_discrete_map=color_map,
                    category_orders={"Result Range": ordered_labels},

                    hover_name="name",
                    hover_data={
                        "year": True,
                        "country": True,
                        "numeric_result": True,
                        "Medal Display": True,
                        "Age": True,
                        "weight": False,
                        "height": False,
                        "medal": False,
                        "nationality": False,
                        "gender": False,
                        "Result Range": False
                    },
                    title=f"Height vs Weight: {selected_gender} - {e_name}",
                    labels={
                        "weight": "Weight (kg)",
                        "height": "Height (cm)",
                        "numeric_result": unit_title,
                        "Medal Display": "Medal",
                        "country": "Country",
                        "Result Range": f"Result ({unit_title})"
                    }
                )

                fig_phys.update_layout(
                    height=600,
                    plot_bgcolor='white',
                    xaxis=dict(showgrid=True, gridcolor='lightgray'),
                    yaxis=dict(showgrid=True, gridcolor='lightgray'),
                    legend=dict(title_font_family="Arial", font=dict(size=10))
                )

                # *** VISUAL CHANGES HERE ***
                # Changed size to 16, and line width to 0 (no border)
                fig_phys.update_traces(marker=dict(size=21, line=dict(width=0)))

                st.plotly_chart(fig_phys, use_container_width=True)
                st.caption(
                    f"🎨 **Color Legend:** Darker shades = Better Results. The ranges represent actual {unit_title.lower()}.")

            else:
                st.info(f"Not enough data available for {selected_gender} in this event since 1960.")
        else:
            st.error("Bio data not available.")
