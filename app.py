from data_processor import *
from data_loader import *
from PIL import Image

pd.set_option('future.no_silent_downcasting', True)

# Load the image
icon_image = Image.open("olympics ring.png")

st.set_page_config(
    layout="wide",
    page_title="Olympics Dashboard",
    page_icon=icon_image  # <--- Use the loaded image variable
)

# Initialize session state for interactive elements
if 'selected_country' not in st.session_state:
    st.session_state.selected_country = 'Israel'

# --- LOAD PROCESSED DATA ---
try:
    data = get_processed_main_data()
    host_data = create_host_advantage_file()
    athletics_df = get_processed_athletics_data()
    gap_df = get_processed_gapminder_data()
    country_ref = load_raw_country_data()
    medals_data = get_processed_medals_data()

    medals_only = data[data['medal'] != 'No medal']
    medals_only = medals_only.drop_duplicates(subset=['year', 'event', 'noc', 'medal'])
    total_medals_per_country = medals_data.groupby('country')['total'].sum().reset_index()
    country_list = sorted(medals_only['team'].dropna().unique())

    # --- SIDEBAR NAVIGATION ---
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
        show_global_overview(medals_only, total_medals_per_country, country_list, medals_data)

    elif page == "🏠 Host Advantage":
        from host_advantage import show_host_advantage
        show_host_advantage(host_data, medals_only, country_ref)

    elif page == "🏃 Athletics Deep Dive":
        from athletics_deep_dive import show_athletics_deep_dive
        show_athletics_deep_dive(athletics_df)

    elif page == "📈 Wellness & Winning Over Time":
        from wellness_winning import show_wellness_winning
        show_wellness_winning(gap_df)

except Exception as e:
    st.error(f"Error loading datasets: {e}")
