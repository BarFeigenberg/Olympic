from data_processor import *
from data_loader import *

st.set_page_config(layout="wide", page_title="Olympics Dashboard")

if 'selected_country' not in st.session_state:
    st.session_state.selected_country = 'Israel'

try:
    data = get_processed_main_data()
    host_data = create_host_advantage_file()
    athletics_df = get_processed_athletics_data()
    gap_df = get_processed_gapminder_data()
    country_ref = load_raw_country_data()
    medals_data = get_processed_medals_data()

    medals_only = data[data['Medal'] != 'No medal']
    map_data = medals_data.groupby('country')['total'].sum().reset_index()
    map_data.rename(columns={'Medal': 'Total Medals'}, inplace=True)
    country_list = sorted(medals_only['Team'].dropna().unique())

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:",
                            ["🌍 Global Overview",
                             "🏠 Host Advantage",
                             "🏃 Athletics Deep Dive",
                             "📈 Wellness & Winning Over Time"],
                            key="main_nav")
    st.sidebar.divider()

    if page == "🌍 Global Overview":
        from global_overview import show_global_overview

        show_global_overview(data, st.session_state.selected_country)

    elif page == "🏠 Host Advantage":
        from host_advantage import show_host_advantage

        show_host_advantage(host_data, data, country_ref)

    elif page == "🏃 Athletics Deep Dive":
        from athletics_deep_dive import show_athletics_deep_dive

        show_athletics_deep_dive(athletics_df)

    elif page == "📈 Wellness & Winning Over Time":
        from wellness_winning import show_wellness_winning

        show_wellness_winning(gap_df)

except Exception as e:
    st.error(f"Error loading datasets: {e}")
