import streamlit as st
import plotly.express as px
import os
import base64


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
    if athletics_df is None or athletics_df.empty:
        st.error("⚠️ Data not found! Please check 'results.csv'.")
        return

    # Data Cleaning
    athletics_df['gender'] = athletics_df['gender'].astype(str).str.strip().str.upper()
    events = sorted(athletics_df['baseevent'].unique().tolist())

    # --- 1. SELECTION FILTERS (NEW LAYOUT: 4, 2, 1, 1) ---
    # Col 1: Title (Wide)
    # Col 2: Spacer (Medium)
    # Col 3: Event Button (Narrow)
    # Col 4: Display Button (Narrow)
    col_title, col_spacer, col_event, col_mode = st.columns([4, 2, 1, 1], gap="small")

    with col_title:
        st.title("🏃 Athletics Analysis")

    with col_spacer:
        st.empty()

    with col_event:
        # Spacers to push the widget down to align with title text
        st.write("")
        st.write("")
        # Default index logic preserved from Raz's code
        default_idx = events.index('100M') if '100M' in events else 0
        e_name = st.selectbox("Select Event:", events, index=default_idx)

    with col_mode:
        # Spacers to push the widget down
        st.write("")
        st.write("")
        mode = st.selectbox("Display:", ['Men Only', 'Women Only', 'Both'])

    # -------------------------------------------------------

    vdf = athletics_df[athletics_df['baseevent'] == e_name].copy()
    if mode == 'Men Only':
        vdf = vdf[vdf['gender'].isin(['M', 'MEN', 'MALE'])]
    elif mode == 'Women Only':
        vdf = vdf[vdf['gender'].isin(['W', 'WOMEN', 'FEMALE'])]

    vdf = vdf.dropna(subset=['numericresult'])

    if vdf.empty:
        st.warning(f"No results found for {e_name}.")
    else:
        # --- 2. RECORD CALCULATIONS ---
        is_high = any(x in e_name for x in ['Throw', 'Jump', 'Vault', 'athlon'])
        unit_title = "Points" if "athlon" in e_name else ("Meters" if is_high else "Seconds")
        if not is_high and vdf['numericresult'].max() > 100:
            unit_title = "Time"

        # Record progression logic (Cumulative Min/Max)
        if is_high:
            yearly_best = vdf.groupby(['year', 'gender'])['numericresult'].max().reset_index()
            yearly_best['running_record'] = yearly_best.groupby('gender')['numericresult'].cummax()
        else:
            yearly_best = vdf.groupby(['year', 'gender'])['numericresult'].min().reset_index()
            yearly_best['running_record'] = yearly_best.groupby('gender')['numericresult'].cummin()

        yearly_best = yearly_best.sort_values('year')
        record_breaks = yearly_best[yearly_best['numericresult'] == yearly_best['running_record']].copy()
        record_breaks = record_breaks.drop_duplicates(subset=['gender', 'numericresult'], keep='first')
        record_breaks = record_breaks.merge(vdf[['year', 'gender', 'numericresult', 'name']],
                                            on=['year', 'gender', 'numericresult'], how='left')

        current_record_holder = record_breaks.sort_values('year').groupby('gender').tail(1)
        historical_record_breaks = record_breaks.drop(current_record_holder.index)

        # --- 3. UPPER SECTION: TREND CHART & PROFILE ---
        col_graph, col_info = st.columns([3, 1])

        with col_graph:
            st.subheader(f"Record Progression: {e_name}")

            # Applying Reference Design: DodgerBlue line color (#1E90FF)
            fig_trend = px.line(
                yearly_best, x='year', y='numericresult', color='gender',
                color_discrete_map={'M': '#1E90FF', 'W': '#1E90FF'}  # Uniform blue lines per reference
            )

            # Style the line and markers
            fig_trend.update_traces(line_width=2, marker_size=8)

            # Add Historical Record Markers (DodgerBlue dots)
            fig_trend.add_scatter(
                x=historical_record_breaks['year'], y=historical_record_breaks['numericresult'],
                mode='markers',
                marker=dict(size=8, color='#1E90FF', line=dict(width=1, color='white')),
                customdata=historical_record_breaks['name'],
                name='Past Record', showlegend=False
            )

            # Add Current Record Highlight (DarkOrange #FF8C00 - same as "Host Year" in reference)
            fig_trend.add_scatter(
                x=current_record_holder['year'], y=current_record_holder['numericresult'],
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

        # --- 4. LOWER SECTION: COUNTRY SCATTER ---
        st.divider()
        st.subheader("Best Result per Country")

        vdf['has_both'] = vdf['country'].map(vdf.groupby('country')['gender'].nunique()) == 2
        best_per_country = vdf.sort_values('numericresult', ascending=not is_high).groupby(
            ['country', 'gender']).first().reset_index()

        ranks = best_per_country.groupby('country').agg(
            {'numericresult': 'max' if is_high else 'min', 'has_both': 'first'}).reset_index()
        ranks = ranks.sort_values(['has_both', 'numericresult'], ascending=[True, True if is_high else False])

        # Using DodgerBlue scatter markers
        fig_country = px.scatter(
            best_per_country, x='numericresult', y="country", color="gender",
            color_discrete_map={'M': '#1E90FF', 'W': '#FF8C00' if mode == 'Both' else '#1E90FF'},
            height=max(500, len(best_per_country) * 20)
        )

        # Highlight Global Best in Orange
        global_best = best_per_country['numericresult'].max() if is_high else best_per_country['numericresult'].min()
        best_rows = best_per_country[best_per_country['numericresult'] == global_best]
        fig_country.add_scatter(
            x=best_rows['numericresult'], y=best_rows['country'],
            mode='markers',
            marker=dict(color='#FF8C00', size=12, line=dict(width=1, color='white')),
            name='All-Time Best', showlegend=True
        )

        fig_country.update_layout(
            plot_bgcolor='white',
            yaxis=dict(categoryorder='array', categoryarray=ranks['country'].tolist(), showgrid=True,
                       gridcolor='lightgray'),
            xaxis=dict(title=unit_title, showgrid=True, gridcolor='lightgray')
        )

        if not is_high:
            fig_country.update_xaxes(autorange="reversed")

        st.plotly_chart(fig_country, width='stretch')
