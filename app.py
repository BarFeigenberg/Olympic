# app.py
# Main entry point for the Streamlit dashboard application
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data_processor import get_processed_main_data, get_processed_host_data, get_processed_athletics_data, \
    get_processed_gapminder_data

st.set_page_config(layout="wide", page_title="Olympics Dashboard")

# Initialize session state for interactive elements
if 'selected_country' not in st.session_state:
    st.session_state.selected_country = 'United States'

# --- LOAD PROCESSED DATA ---
try:
    data = get_processed_main_data()
    host_data = get_processed_host_data()
    athletics_df = get_processed_athletics_data()
    gap_df = get_processed_gapminder_data()

    medals_only = data[data['Medal'] != 'No medal']
    map_data = medals_only.groupby('country')['Medal'].count().reset_index()
    map_data.rename(columns={'Medal': 'Total Medals'}, inplace=True)
    country_list = sorted(map_data['country'].dropna().unique())

    # --- SIDEBAR NAVIGATION ---
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:",
                            ["🌍 Global Overview",
                             "🏠 Host Advantage",
                             "🏃 Athletics Deep Dive",
                             "📈 Wellness & Winning Over Time"])
    st.sidebar.divider()

    if page == "🌍 Global Overview":

        st.title("🏅 Global Olympic Insights")
        st.divider()

        # ROW 1: CONTROLS
        col_select, col_metrics = st.columns([1, 3], gap="medium")
        with col_select:
            if st.session_state.selected_country not in country_list:
                st.session_state.selected_country = country_list[0] if country_list else "USA"

            selected_country = st.selectbox(
                "Select Country:",
                country_list,
                index=country_list.index(st.session_state.selected_country)
            )
            if selected_country != st.session_state.selected_country:
                st.session_state.selected_country = selected_country
                st.rerun()

        with col_metrics:
            country_df = medals_only[medals_only['country'] == st.session_state.selected_country]
            total = len(country_df)
            best_sport = country_df['Sport'].mode()[0] if not country_df.empty else "N/A"
            best_year = country_df['Year'].mode()[0] if not country_df.empty else "N/A"

            m1, m2, m3 = st.columns(3)
            m1.metric("🥇 Total Medals", total)
            m2.metric("🏆 Best Sport", best_sport)
            m3.metric("📅 Best Year", str(best_year))

        st.divider()

        # ROW 2: MAP
        pastel_scale = [[0.0, "#C1E1C1"], [0.5, "#FDFD96"], [1.0, "#FF6961"]]
        fig_map = px.choropleth(
            map_data,
            locations="country",
            locationmode="country names",
            color="Total Medals",
            hover_name="country",
            custom_data=["country"],
            color_continuous_scale=pastel_scale,
            projection="natural earth"
        )

        sel_data = map_data[map_data['country'] == st.session_state.selected_country]
        if not sel_data.empty:
            fig_map.add_trace(go.Choropleth(
                locations=sel_data['country'],
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

        # ROW 3: GRAPHS
        c_trend, c_podium = st.columns(2, gap="large")
        with c_trend:
            st.subheader("📈 Medals Trend")
            td = country_df.groupby('Year')['Medal'].count().reset_index()
            if not td.empty:
                fig = px.area(td, x='Year', y='Medal', markers=True, color_discrete_sequence=["#77DD77"])
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

        with c_podium:
            st.subheader("🏆 Top 3 Sports Podium")
            top = country_df.groupby('Sport')['Medal'].count().reset_index().sort_values('Medal', ascending=False).head(3).reset_index(drop=True)
            if len(top) >= 1:
                pod_data = [{'Sport': top.iloc[0]['Sport'], 'Medal': top.iloc[0]['Medal'], 'Color': '#FFD700', 'Pos': 2}]
                if len(top) >= 2: pod_data.append({'Sport': top.iloc[1]['Sport'], 'Medal': top.iloc[1]['Medal'], 'Color': '#C0C0C0', 'Pos': 1})
                if len(top) >= 3: pod_data.append({'Sport': top.iloc[2]['Sport'], 'Medal': top.iloc[2]['Medal'], 'Color': '#CD7F32', 'Pos': 3})
                pdf = pd.DataFrame(pod_data).sort_values('Pos')
                fig = go.Figure(go.Bar(x=pdf['Sport'], y=pdf['Medal'], marker_color=pdf['Color'], text=pdf['Medal'], textposition='auto'))
                fig.update_layout(height=400, yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)

    # --- PAGE 2: HOST IMPACT ---
    elif page == "🏠 Host Advantage":
        st.title("🏠 Host Advantage Effect")
        st.divider()
        host_data['Label'] = host_data['Year'].astype(str) + " - " + host_data['Host_City'] + " (" + host_data['Host_NOC'] + ")"
        options = sorted(host_data['Label'].unique(), reverse=True)
        sel_event = st.selectbox("Select Host Event:", options)

        if sel_event:
            row = host_data[host_data['Label'] == sel_event].iloc[0]
            h_year, h_noc, h_medals = int(row['Year']), row['Host_NOC'], int(row['Total_Medals'])
            past = medals_only[(medals_only['NOC'] == h_noc) & (medals_only['Year'] < h_year)]
            avg = past.groupby('Year')['Medal'].count().mean() if not past.empty else 0.0
            c1, c2 = st.columns([1, 2], gap="large")
            with c1:
                st.metric("Medals (Host Year)", h_medals)
                st.metric("Historical Avg", f"{avg:.1f}")
                if avg > 0:
                    boost = ((h_medals - avg) / avg) * 100
                    st.metric("Boost", f"{boost:.1f}%", delta=f"{boost:.1f}%")
                else: st.metric("Boost", "N/A")
            with c2:
                comp = pd.DataFrame({'Type': ['Avg', 'Host'], 'Medals': [avg, h_medals]})
                fig = px.bar(comp, x='Type', y='Medals', text='Medals', color='Type', color_discrete_sequence=['#BDC3C7', '#FF6961'])
                st.plotly_chart(fig, use_container_width=True)
    # ==================================================
    # PAGE 3: ATHLETICS DEEP DIVE (Time Format Fix)
    # ==================================================
    elif page == "🏃 Athletics Deep Dive":
        st.title("🏃 Athletics Analysis")

        if athletics_df is not None and not athletics_df.empty:
            # Clean gender strings
            athletics_df['Gender'] = athletics_df['Gender'].astype(str).str.strip().str.upper()

            events = sorted(athletics_df['BaseEvent'].unique().tolist())
            e_name = st.selectbox("Select Event:", events)
            mode = st.selectbox("Display:", ['Men Only', 'Women Only', 'Both'])

            # Filter by Event
            vdf = athletics_df[athletics_df['BaseEvent'] == e_name].copy()

            # Apply Gender Filter
            if mode == 'Men Only':
                vdf = vdf[vdf['Gender'].isin(['M', 'MEN', 'MALE'])]
            elif mode == 'Women Only':
                vdf = vdf[vdf['Gender'].isin(['W', 'WOMEN', 'FEMALE'])]

            # Remove invalid results
            vdf = vdf.dropna(subset=['NumericResult'])

            if vdf.empty:
                st.warning(f"No valid numeric data found for {e_name} in {mode} mode.")
                if st.checkbox("Show raw data sample for debugging"):
                    st.write(athletics_df[athletics_df['BaseEvent'] == e_name].head(10))
            else:
                # 1. Determine Event Type (Time vs. Distance/Points)
                # 'is_high' means higher number is better (Jumps, Throws, Points)
                # If False, it implies a Time event (Running, Walking) where lower is better
                is_high = any(x in e_name for x in ['Throw', 'Jump', 'Vault', 'athlon'])

                # 2. Prepare Data for Plotting
                vdf['Has_Both'] = vdf['Country'].map(vdf.groupby('Country')['Gender'].nunique()) == 2

                # Sort to find best result per country/gender
                # If is_high (Jump) -> Descending (Max). If Time -> Ascending (Min).
                best = vdf.sort_values('NumericResult', ascending=not is_high).groupby(
                    ['Country', 'Gender']).first().reset_index()

                # Calculate sorting order for the categorical Y-axis (Countries)
                ranks = best.groupby('Country').agg(
                    {'NumericResult': 'max' if is_high else 'min', 'Has_Both': 'first'}).reset_index()
                ranks = ranks.sort_values(['Has_Both', 'NumericResult'],
                                          ascending=[True, True if is_high else False])

                # 3. Handle Axis Formatting Logic
                x_axis_column = "NumericResult"  # Default
                x_title = "Result"
                tick_format = None  # Default auto formatting

                if is_high:
                    # Logic for Distance/Points
                    if "athlon" in e_name:
                        x_title = "Points"
                    else:
                        x_title = "Meters"
                else:
                    # Logic for Time events
                    max_time = best['NumericResult'].max()

                    # Threshold: If max time is > 100 seconds, switch to Time Format
                    # Otherwise (sprints), keep as Seconds (10.5s is easier to read than 00:00:10.5)
                    if max_time > 100:
                        # Convert seconds to a datetime object (using a dummy reference date)
                        # This allows Plotly to handle the axis as Time
                        base_date = pd.Timestamp("1970-01-01")

                        # Create a new column for the plot X-axis
                        best['Time_Axis'] = best['NumericResult'].apply(
                            lambda x: base_date + pd.Timedelta(seconds=x)
                        )
                        x_axis_column = "Time_Axis"

                        # Determine formatting based on duration (Hours vs Minutes)
                        if max_time > 3600:  # More than 1 hour (Marathon, 50km walk)
                            tick_format = "%H:%M:%S"
                            x_title = "Hours"
                        else:  # Between 100s and 1 hour (800m, 1500m, 5k, 10k)
                            tick_format = "%M:%S"
                            x_title = "Minutes"
                    else:
                        # Sprints (100m, 200m, 400m)
                        x_title = "Seconds"

                # 4. Generate Plot
                fig = px.scatter(
                    best,
                    x=x_axis_column,
                    y="Country",
                    color="Gender",
                    color_discrete_map={'M': '#1f77b4', 'W': '#e377c2'},
                    height=max(500, len(best) * 20),
                    hover_name="Country",
                    # Pass both raw result and formatted result to hover data
                    hover_data={'Result': True, 'Year': True, 'Name': True, x_axis_column: False}
                )

                # Apply layout updates
                layout_args = dict(
                    yaxis=dict(categoryorder='array', categoryarray=ranks['Country'].tolist()),
                    xaxis_title=x_title
                )

                # Apply specific time format if needed
                if tick_format:
                    layout_args['xaxis'] = dict(tickformat=tick_format)

                fig.update_layout(**layout_args)
                st.plotly_chart(fig, use_container_width=True)
    # --- PAGE 4: WELLNESS & WINNING ---
    elif page == "📈 Wellness & Winning Over Time":
        st.title("📈 Wellness & Winning Over Time")
        if gap_df is not None and not gap_df.empty:
            roi_mode = st.radio("Select View:", ["Efficiency (Medals per Million)", "Total Impact (Total Medals)"], horizontal=True)
            if roi_mode == "Efficiency (Medals per Million)":
                fig = px.scatter(gap_df, x="Life_Expectancy", y="Medals_Per_Million", animation_frame="Year", animation_group="Country_Name", size="Delegation_Size", color="continent", hover_name="Country_Name", size_max=50, range_x=[35, 90], range_y=[-0.5, 6])
            else:
                gap_df['Medals_Log_Proxy'] = gap_df['Medals'].replace(0, 0.5)
                fig = px.scatter(gap_df, x="Life_Expectancy", y="Medals_Log_Proxy", animation_frame="Year", animation_group="Country_Name", size="Delegation_Size", color="continent", hover_name="Country_Name", hover_data=["Medals"], size_max=50, range_x=[35, 90], range_y=[0.4, 450], log_y=True)
            fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 700
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Gapminder/ROI data could not be loaded.")

except Exception as e:
    st.error(f"Application Error: {e}")