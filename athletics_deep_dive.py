
import base64
from data_processor import *


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


def format_time_value(seconds):
    """Convert seconds to appropriate time format string."""
    if seconds >= 3600:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}:{mins:02d}:{secs:05.2f}"
    elif seconds >= 60:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}:{secs:05.2f}"
    else:
        return f"{seconds:.2f}"


def filter_time_outliers(df, result_col='numeric_result'):
    """
    Filter out outliers in time-based events using median-based method.
    This removes garbage data from old records that were entered incorrectly.
    
    Uses median as the reference point - keeps only values within 0.5x to 2x the median.
    This is more robust than IQR when dealing with extreme outliers.
    """
    median_val = df[result_col].median()
    
    # For time-based events (lower is better), reasonable times are:
    # - Minimum: at least 50% of median (no one is THAT much faster)
    # - Maximum: at most 200% of median (outliers are typically 10-20x higher)
    lower_bound = median_val * 0.5
    upper_bound = median_val * 2.0
    
    # Filter: keep only values within reasonable bounds
    filtered = df[(df[result_col] >= lower_bound) & (df[result_col] <= upper_bound)].copy()
    
    return filtered


def get_time_unit_info(data_series):
    """Determine the best time unit based on MEDIAN value in data (avoids outliers)."""
    median_seconds = data_series.median()
    if median_seconds >= 3600:
        return "Hours", "h", 3600
    elif median_seconds >= 60:
        return "Minutes", "min", 60
    else:
        return "Seconds", "s", 1


def convert_to_display_unit(seconds, divisor):
    """Convert seconds to display unit (minutes or hours)."""
    return seconds / divisor


def show_athletics_deep_dive(athletics_df, country_ref):
    bio_df = load_athlete_bio_data()

    if athletics_df is None or athletics_df.empty:
        st.error("Data not found! Please check 'results.csv'.")
        return
    if country_ref is None or country_ref.empty:
        st.error("Data not found! Please check 'Olympics_Country.csv'.")
        return

    # Data Cleaning
    athletics_df.columns = athletics_df.columns.str.lower()
    athletics_df = athletics_df.rename(columns={'numericresult': 'numeric_result'})

    if 'year' in athletics_df.columns:
        athletics_df = athletics_df[pd.to_numeric(athletics_df['year'], errors='coerce').notna()]
        athletics_df['year'] = athletics_df['year'].astype(int)

    if 'event' in athletics_df.columns:
        athletics_df['event'] = athletics_df['event'].astype(str).str.strip().str.upper()
        athletics_df['event'] = athletics_df['event'].str.replace(' MEN', '', regex=False)
        athletics_df['event'] = athletics_df['event'].str.replace(' WOMEN', '', regex=False)
        athletics_df['event'] = athletics_df['event'].str.strip()

    athletics_df = athletics_df.dropna(subset=['numeric_result'])

    if not athletics_df.empty:
        exclude_keywords = ['10 MILE', '5 MILES', '3KM WALK', '3500M WALK', 'STANDING', '10KM', '80M HURDLES']
        athletics_df = athletics_df[
            ~athletics_df['event'].str.contains('|'.join(exclude_keywords), case=False, na=False)]

        event_counts = athletics_df['event'].value_counts()
        valid_events = event_counts[event_counts >= 5].index
        athletics_df = athletics_df[athletics_df['event'].isin(valid_events)]

        if not athletics_df.empty:
            event_max_years = athletics_df.groupby('event')['year'].max()
            modern_events = event_max_years[event_max_years > 1950].index
            athletics_df = athletics_df[athletics_df['event'].isin(modern_events)]

    athletics_df['gender'] = athletics_df['gender'].astype(str).str.strip().str.upper()
    events = sorted(athletics_df['event'].unique().tolist())

    # --- 1. TITLE SECTION ---
    st.title("Athletics Analysis")
    st.divider()

    # --- 2. EVENT SELECTION ---
    # CSS for tight alignment and vertical lift
    st.markdown("""
            <style>
                /* Lift the radio buttons up to match subheader height */
                div.row-widget.stRadio > div {
                    flex-direction: row;
                    justify-content: flex-end;
                    margin-top: -42px !important; 
                }
                /* Make the radio buttons more compact */
                div.row-widget.stRadio div[role="radiogroup"] label {
                    padding: 2px 10px !important;
                    background-color: #f8f9fb;
                    border-radius: 5px;
                    margin-left: 5px !important;
                }
            </style>
        """, unsafe_allow_html=True)
    col_sel, col_empty = st.columns([1, 2])
    with col_sel:
        default_idx = events.index('100M') if '100M' in events else 0
        e_name = st.selectbox("Select Event:", events, index=default_idx)

    vdf = athletics_df[athletics_df['event'] == e_name].copy()
    vdf['gender_label'] = vdf['gender'].map({
        'M': 'Men', 'MEN': 'Men', 'MALE': 'Men',
        'W': 'Women', 'WOMEN': 'Women', 'FEMALE': 'Women'
    }).fillna(vdf['gender'])
    vdf = vdf.dropna(subset=['numeric_result'])

    if vdf.empty:
        st.warning(f"No results found for {e_name}.")
        return

    # --- 3. PHYSICAL ANALYSIS ---
    st.divider()

    # Title and Gender Selection on the same line
    col_title_phys, col_radio_phys = st.columns([2, 1])

    with col_title_phys:
        st.subheader(f"Physical Analysis - {e_name}")

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

        if not merged.empty:
            gender_options = sorted(merged['gender_label'].unique())

            with col_radio_phys:
                # The radio now sits high and tight thanks to the CSS above
                selected_gender = st.radio(
                    "Select Gender:",
                    gender_options,
                    horizontal=True,
                    label_visibility="collapsed"
                )

            merged_filtered = merged[merged['gender_label'] == selected_gender].copy()

            if not merged_filtered.empty:
                # Unit Setup - Dynamic time conversion
                if "ATHLON" in e_name:
                    unit_title, y_suffix, is_high = "Points", " pts", True
                    time_divisor = 1
                elif any(x in e_name for x in ['THROW', 'JUMP', 'VAULT', 'PUT']):
                    unit_title, y_suffix, is_high = "Meters", "m", True
                    time_divisor = 1
                else:
                    # Time-based events: FILTER OUTLIERS first, then determine best unit
                    merged_filtered = filter_time_outliers(merged_filtered, 'numeric_result')
                    if merged_filtered.empty:
                        st.warning("No valid data after filtering outliers.")
                        return
                    unit_title, y_suffix, time_divisor = get_time_unit_info(merged_filtered['numeric_result'])
                    is_high = False
                    # Convert values to display unit
                    merged_filtered['display_result'] = merged_filtered['numeric_result'] / time_divisor

                # Country Mapping & Colors
                noc_map_local = dict(
                    zip(country_ref['noc'], country_ref['country'])) if not country_ref.empty else {}
                merged_filtered['country_full'] = merged_filtered['country'].map(noc_map_local).fillna(
                    merged_filtered['country'])

                blue_shades = ['#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c',
                               '#08306b']
                pink_shades = ['#ffe8eb', '#fde0dd', '#fcc5c0', '#fa9fb5', '#f768a1', '#dd3497', '#ae017e',
                               '#7a0177']
                current_shades = pink_shades if 'Women' in str(selected_gender) else blue_shades

                # Use display_result for time events, numeric_result for others
                result_col = 'display_result' if time_divisor > 1 else 'numeric_result'

                # Legend sorting logic
                try:
                    merged_filtered['tier_interval'] = pd.qcut(merged_filtered[result_col], q=8,
                                                               duplicates='drop')
                except ValueError:
                    merged_filtered['tier_interval'] = pd.qcut(merged_filtered[result_col], q=4,
                                                               duplicates='drop')

                unique_intervals = sorted(merged_filtered['tier_interval'].unique())
                num_tiers = len(unique_intervals)
                label_map, color_map, ordered_labels = {}, {}, []

                for i, interval in enumerate(unique_intervals):
                    label_text = f"{interval.left:.2f} - {interval.right:.2f}"
                    ordered_labels.append(label_text)
                    color_idx = int((i / (num_tiers - 1)) * 7) if is_high else int(
                        ((num_tiers - 1 - i) / (num_tiers - 1)) * 7)
                    color_map[label_text] = current_shades[color_idx]
                    label_map[interval] = label_text

                merged_filtered['Result Range'] = merged_filtered['tier_interval'].map(label_map)
                ordered_labels = list(reversed(ordered_labels))

                # Chart Creation
                fig_phys = px.scatter(
                    merged_filtered, x="weight", y="height", color="Result Range",
                    color_discrete_map=color_map, category_orders={"Result Range": ordered_labels},
                    hover_name="name",
                    hover_data={"year": True, "country_full": True, "numeric_result": True, "Age": True,
                                "weight": False, "height": False, "country": False, "Result Range": False},
                    labels={"weight": "Weight (kg)", "height": "Height (cm)", "country_full": "Country",
                            "Result Range": f"Result ({unit_title})"}
                )
                fig_phys.update_layout(height=600, plot_bgcolor='white',
                                       margin=dict(t=10))  # Small top margin since title is outside
                fig_phys.update_traces(marker=dict(size=20, opacity=0.8))

                st.plotly_chart(fig_phys, width='stretch')
                st.caption(f"**Color Legend:** Darker shades = Better Results.")
        else:
            st.info("Not enough physical data available.")

    # --- 4. RECORD PROGRESSION (NOW LAST) ---
    st.divider()
    st.subheader(f"Record Progression: {e_name}")

    # Re-calculate units for records - Dynamic time conversion
    if "ATHLON" in e_name:
        unit_title, y_suffix, is_high = "Points", " pts", True
        time_divisor = 1
        vdf_clean = vdf.copy()
    elif any(x in e_name for x in ['THROW', 'JUMP', 'VAULT', 'PUT']):
        unit_title, y_suffix, is_high = "Meters", "m", True
        time_divisor = 1
        vdf_clean = vdf.copy()
    else:
        # Time-based events: FILTER OUTLIERS first
        vdf_clean = filter_time_outliers(vdf, 'numeric_result')
        if vdf_clean.empty:
            st.warning("No valid data after filtering outliers for Record Progression.")
            return
        unit_title, y_suffix, time_divisor = get_time_unit_info(vdf_clean['numeric_result'])
        is_high = False

    if is_high:
        yearly_best = vdf_clean.groupby(['year', 'gender_label'])['numeric_result'].max().reset_index()
        yearly_best['running_record'] = yearly_best.groupby('gender_label')['numeric_result'].cummax()
    else:
        yearly_best = vdf_clean.groupby(['year', 'gender_label'])['numeric_result'].min().reset_index()
        yearly_best['running_record'] = yearly_best.groupby('gender_label')['numeric_result'].cummin()

    # Convert to display units for time-based events
    if time_divisor > 1:
        yearly_best['display_result'] = yearly_best['numeric_result'] / time_divisor
        y_col = 'display_result'
    else:
        y_col = 'numeric_result'

    yearly_best = yearly_best.sort_values('year')
    record_breaks = yearly_best[yearly_best['numeric_result'] == yearly_best['running_record']].copy()
    record_breaks = record_breaks.drop_duplicates(subset=['gender_label', 'numeric_result'], keep='first')
    record_breaks = record_breaks.merge(vdf_clean[['year', 'gender_label', 'numeric_result', 'name', 'country']],
                                        on=['year', 'gender_label', 'numeric_result'], how='left')

    # Add display_result to record_breaks too
    if time_divisor > 1:
        record_breaks['display_result'] = record_breaks['numeric_result'] / time_divisor

    current_record_holder = record_breaks.sort_values('year').groupby('gender_label').tail(1)
    historical_record_breaks = record_breaks.drop(current_record_holder.index)

    men_color, women_color = '#1E90FF', 'violet'
    fig_trend = px.line(yearly_best, x='year', y=y_col, color='gender_label',
                        color_discrete_map={'Men': men_color, 'Women': women_color})

    # Define Darker Shades for the Current Record Markers
    dark_men = '#0047AB'  # Cobalt Blue
    dark_women = '#C71585'  # Medium Violet Red

    def add_record_trace(df, color, default_size, is_current, group_name):
        if df.empty: return

        c_data = df[['name', 'country']].fillna('N/A').values

        # Use display_result for time-based events
        trace_y_col = 'display_result' if time_divisor > 1 else 'numeric_result'

        # Logic for Current Record color: Darker version of the gender color
        if is_current:
            marker_color = dark_men if group_name == 'Men' else dark_women
            marker_size = 12
        else:
            marker_color = color
            marker_size = default_size

        fig_trend.add_scatter(
            x=df['year'],
            y=df[trace_y_col],
            mode='markers',
            marker=dict(
                size=marker_size,
                color=marker_color,
                line=dict(width=1, color='white')
            ),
            customdata=c_data,
            hovertemplate=(
                    "<b>%{customdata[0]}</b> (%{customdata[1]})<br>" +
                    "Year: %{x}<br>" +
                    f"Result: %{{y:.2f}}{y_suffix}<br>" +
                    "<extra></extra>"
            ),
            name=f"Current Record ({group_name})" if is_current else group_name,
            legendgroup=group_name,
            showlegend=False
        )

    add_record_trace(historical_record_breaks[historical_record_breaks['gender_label'] == 'Men'], men_color, 9, False,
                     'Men')
    add_record_trace(current_record_holder[current_record_holder['gender_label'] == 'Men'], men_color, 15, True, 'Men')
    add_record_trace(historical_record_breaks[historical_record_breaks['gender_label'] == 'Women'], women_color, 9,
                     False, 'Women')
    add_record_trace(current_record_holder[current_record_holder['gender_label'] == 'Women'], women_color, 15, True,
                     'Women')

    # 1. Use unique years present in data to avoid gaps (1940-1944 etc.)
    tick_vals = sorted(vdf_clean['year'].unique())

    # 2. Create labels with "OR" for record years
    tick_text = []
    for y in tick_vals:
        is_men_rec = not current_record_holder[
            (current_record_holder['year'] == y) & (current_record_holder['gender_label'] == 'Men')].empty
        is_women_rec = not current_record_holder[
            (current_record_holder['year'] == y) & (current_record_holder['gender_label'] == 'Women')].empty

        if is_men_rec and is_women_rec:
            tick_text.append(
                f"<b>{y}</b><br><span style='color:{dark_men}'>Olympic Record</span> <span style='color:{dark_women}'>OR</span>")
        elif is_men_rec:
            tick_text.append(f"<b>{y}</b><br><span style='color:{dark_men}'>Olympic Record</span>")
        elif is_women_rec:
            tick_text.append(f"<b>{y}</b><br><span style='color:{dark_women}'>Olympic Record</span>")
        else:
            tick_text.append(str(y))

    # Calculate the yearly best results with names for hover
    if is_high:
        yearly_best_hover = vdf_clean.groupby(['year', 'gender_label'])['numeric_result'].max().reset_index()
    else:
        yearly_best_hover = vdf_clean.groupby(['year', 'gender_label'])['numeric_result'].min().reset_index()

    yearly_best_hover = yearly_best_hover.merge(
        vdf_clean[['year', 'gender_label', 'numeric_result', 'name', 'country']],
        on=['year', 'gender_label', 'numeric_result'],
        how='left').drop_duplicates(subset=['year', 'gender_label'])  # Ensure one row per year/gender

    # Add display_result for time-based events
    if time_divisor > 1:
        yearly_best_hover['display_result'] = yearly_best_hover['numeric_result'] / time_divisor
        hover_y_col = 'display_result'
    else:
        hover_y_col = 'numeric_result'

    # Create the base figure
    fig_trend = px.line(yearly_best_hover, x='year', y=hover_y_col, color='gender_label',
                        color_discrete_map={'Men': men_color, 'Women': women_color})

    # Update each gender trace separately to ensure correct hover alignment
    for gender in yearly_best_hover['gender_label'].unique():
        # Filter data for this specific gender and sort by year
        gender_data = yearly_best_hover[yearly_best_hover['gender_label'] == gender].sort_values('year')

        # Prepare the specific hover package for this gender
        # Now 'name' and 'country' are safely in the index
        specific_hover = gender_data[['year', 'name', 'country']].fillna('N/A')

        # Apply to the matching trace
        fig_trend.update_traces(
            customdata=specific_hover,
            hovertemplate=(
                    "<b>%{customdata[1]}</b> (%{customdata[2]})<br>" +
                    "Year: %{customdata[0]}<br>" +
                    "Result: %{y:.2f}" + y_suffix +
                    "<extra></extra>"
            ),
            selector={"name": gender}  # Targets ONLY the specific gender line
        )

    for gender in yearly_best_hover['gender_label'].unique():
        # Get the single latest record for this gender
        current_rec = current_record_holder[current_record_holder['gender_label'] == gender]

        if not current_rec.empty:
            # Define the color based on gender
            marker_color = dark_men if gender == 'Men' else dark_women

            # Use display_result if time-based
            rec_y_val = current_rec['display_result'] if time_divisor > 1 else current_rec['numeric_result']

            fig_trend.add_scatter(
                x=current_rec['year'],
                y=rec_y_val,
                mode='markers',
                marker=dict(
                    size=12,
                    color=marker_color,
                ),
                # Important: Match the hover data for the record point too
                customdata=current_rec[['year', 'name', 'country']].fillna('N/A'),
                hovertemplate=(
                        "<b>CURRENT RECORD: %{customdata[1]}</b> (%{customdata[2]})<br>" +
                        "Year: %{customdata[0]}<br>" +
                        "Result: %{y:.2f}" + y_suffix +
                        "<extra></extra>"
                ),
                name=f"Current Record ({gender})",
                showlegend=False
            )

    logical_order = [str(y) for y in sorted(vdf_clean['year'].unique())]
    fig_trend.update_layout(
        height=600,
        plot_bgcolor='white',
        xaxis=dict(
            title="Year",
            type='category',
            # FIX: Explicitly tell Plotly the correct order of years
            categoryorder='array',
            categoryarray=logical_order,
            tickmode='array',
            tickvals=tick_vals,
            ticktext=tick_text,
            tickangle=-270,
            showgrid=False
        ),
        yaxis=dict(
            title=f"Result ({unit_title})",
            ticksuffix=y_suffix,
            showgrid=True,
            gridcolor='whitesmoke'
        )
    )
    st.plotly_chart(fig_trend, width='stretch')