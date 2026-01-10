import os
import base64
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

    if athletics_df is None or athletics_df.empty:
        st.error("⚠️ Data not found! Please check 'results.csv'.")
        return
    print("Check 1")
    # Data Cleaning
    athletics_df.columns = athletics_df.columns.str.lower()
    athletics_df = athletics_df.rename(columns={'numericresult': 'numeric_result'})

    # Filter out embedded header rows (where year is 'Year')
    if 'year' in athletics_df.columns:
        athletics_df = athletics_df[pd.to_numeric(athletics_df['year'], errors='coerce').notna()]
        athletics_df['year'] = athletics_df['year'].astype(int)

    if 'event' in athletics_df.columns:
        # Standardize: UPPERCASE
        athletics_df['event'] = athletics_df['event'].astype(str).str.strip().str.upper()
    # Remove Gender Suffixes to merge "100M Men" and "100m" -> "100M"
        athletics_df['event'] = athletics_df['event'].str.replace(' MEN', '', regex=False)
        athletics_df['event'] = athletics_df['event'].str.replace(' WOMEN', '', regex=False)
        # Handle '100M' vs '100 M' spacing if any
        athletics_df['event'] = athletics_df['event'].str.strip()

    # Filter out rows with no results BEFORE creating the event list
    athletics_df = athletics_df.dropna(subset=['numeric_result'])

    # Filter out events with only 1 result (sparse data, e.g. '10 MILE RACE WALK')
    if not athletics_df.empty:
        # 1. Explicit exclusion of known problematic/historical events
        exclude_keywords = ['10 MILE', '5 MILES', '3KM WALK', '3500M WALK', 'STANDING', '10KM', '80M HURDLES']
        athletics_df = athletics_df[~athletics_df['event'].str.contains('|'.join(exclude_keywords), case=False, na=False)]

        # 2. Minimum data count threshold (Increased to 5 to be safer)
        event_counts = athletics_df['event'].value_counts()
        valid_events = event_counts[event_counts >= 5].index
        athletics_df = athletics_df[athletics_df['event'].isin(valid_events)]

        # 3. Filter out events that ended before 1950 (Archived/Discontinued events)
        if not athletics_df.empty:
            event_max_years = athletics_df.groupby('event')['year'].max()
            modern_events = event_max_years[event_max_years > 1950].index
            athletics_df = athletics_df[athletics_df['event'].isin(modern_events)]

    athletics_df['gender'] = athletics_df['gender'].astype(str).str.strip().str.upper()
    events = sorted(athletics_df['event'].unique().tolist())
    print("Check 4")
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
        vdf = athletics_df[athletics_df['event'] == e_name].copy()

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

        # --- 3. UPPER SECTION: TREND CHART (STRETCHED & CLEAN LAYOUT) ---
        st.subheader(f"Record Progression: {e_name}")

        # Colors
        men_color = '#1E90FF'
        women_color = '#FF69B4'
        color_map = {'Men': men_color, 'Women': women_color}
        record_fill = '#FFD700'
        record_border = '#B8860B'

        # 1. Base Line Chart
        fig_trend = px.line(
            yearly_best,
            x='year',
            y='numeric_result',
            color='gender_label',
            color_discrete_map=color_map,
            labels={'gender_label': 'Gender'}
        )
        fig_trend.update_traces(line_width=2, marker_size=0)
        fig_trend.update_traces(hoverinfo='skip', hovertemplate=None, selector=dict(type='scatter', mode='lines'))

        # 2. Add Markers (Linked to Legend)

        # --- MEN ---
        men_hist = historical_record_breaks[historical_record_breaks['gender_label'] == 'Men']
        men_curr = current_record_holder[current_record_holder['gender_label'] == 'Men']

        fig_trend.add_scatter(
            x=men_hist['year'], y=men_hist['numeric_result'],
            mode='markers',
            marker=dict(size=9, color=men_color, line=dict(width=1, color='white')),
            customdata=men_hist['name'],
            name='Men', legendgroup='Men', showlegend=False
        )
        fig_trend.add_scatter(
            x=men_curr['year'], y=men_curr['numeric_result'],
            mode='markers',
            marker=dict(size=15, color=record_fill, line=dict(width=2, color=record_border)),
            customdata=men_curr['name'],
            name='Men', legendgroup='Men', showlegend=False
        )

        # --- WOMEN ---
        women_hist = historical_record_breaks[historical_record_breaks['gender_label'] == 'Women']
        women_curr = current_record_holder[current_record_holder['gender_label'] == 'Women']

        fig_trend.add_scatter(
            x=women_hist['year'], y=women_hist['numeric_result'],
            mode='markers',
            marker=dict(size=9, color=women_color, line=dict(width=1, color='white')),
            customdata=women_hist['name'],
            name='Women', legendgroup='Women', showlegend=False
        )
        fig_trend.add_scatter(
            x=women_curr['year'], y=women_curr['numeric_result'],
            mode='markers',
            marker=dict(size=15, color=record_fill, line=dict(width=2, color=record_border)),
            customdata=women_curr['name'],
            name='Women', legendgroup='Women', showlegend=False
        )

        # 3. Add Instruction Annotation (ABOVE the Legend)
        # Positioned high up on the right (y=1.35) so it doesn't overlap the items
        fig_trend.add_annotation(
            text="💡 Click legend items to filter gender",
            xref="paper", yref="paper",
            x=1, y=1.35,
            showarrow=False,
            font=dict(size=11, color="gray"),
            xanchor="right"
        )
        # 4. Smart X-Axis Formatting
        min_year = int(vdf['year'].min())
        max_year = 2024
        timeline_years = [y for y in range(1896, max_year + 4, 4) if y >= min_year - 4]

        men_record_year = men_curr['year'].values[0] if not men_curr.empty else None
        women_record_year = women_curr['year'].values[0] if not women_curr.empty else None

        tick_text = []
        for y in timeline_years:
            is_men_rec = (y == men_record_year)
            is_women_rec = (y == women_record_year)

            # Helper for "Olympic RECORD"
            wr_label = '<br><span style="font-size:9px">OLYMPIC RECORD</span>'

            if is_men_rec and is_women_rec:
                tick_text.append(f'<span style="color:purple; font-weight:900;">{y}{wr_label}</span>')
            elif is_men_rec:
                tick_text.append(f'<span style="color:{men_color}; font-weight:900;">{y}{wr_label}</span>')
            elif is_women_rec:
                tick_text.append(f'<span style="color:{women_color}; font-weight:900;">{y}{wr_label}</span>')
            else:
                tick_text.append(str(y))

        fig_trend.update_layout(
            height=500,
            plot_bgcolor='white',
            # Margins: l=0 pushes it to the far left. t=100 gives room for the top legend + text.
            margin=dict(l=0, r=0, t=100, b=0),
            xaxis=dict(
                tickmode='array', tickvals=timeline_years, ticktext=tick_text,
                gridcolor='white', showline=True, linecolor='lightgray',
                range=[min_year - 2, max_year + 2]  # Slight padding on sides
            ),
            yaxis=dict(
                title=unit_title, showgrid=True, gridwidth=1, gridcolor='lightgray',
                rangemode="tozero" if is_high else "normal",
                automargin=True  # Ensures the Y-label doesn't get cut off despite l=0
            ),
            hovermode="closest",
            # Legend position (Top Right, slightly below the instruction text)
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1)
        )

        if not is_high:
            fig_trend.update_yaxes(autorange="reversed")

        st.plotly_chart(fig_trend, width='stretch')

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

                st.plotly_chart(fig_phys, width='stretch')
                st.caption(
                    f"🎨 **Color Legend:** Darker shades = Better Results. The ranges represent actual {unit_title.lower()}.")

            else:
                st.info(f"Not enough data available for {selected_gender} in this event since 1960.")
        else:
            st.error("Bio data not available.")
