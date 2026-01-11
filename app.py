import streamlit as st
from streamlit_option_menu import option_menu
from data_processor import *
from data_loader import *
from PIL import Image

pd.set_option('future.no_silent_downcasting', True)

# Load the image
icon_image = Image.open("olympics ring.png")

# --- 1. CONFIGURATION (Must be the first Streamlit command) ---
st.set_page_config(
    layout="wide",
    page_title="Olympics Dashboard",
    page_icon=icon_image
)


# --- 2. CUSTOM CSS FOR BETTER UI ---
def local_css():
    st.markdown("""
    <style>
        /* Change the background color of the main app area */
        .stApp {
            background-color: #F8FAFC;
        }
    
        /* Ensure the header/toolbar doesn't stay white */
        header[data-testid="stHeader"] {
            background-color: #F8FAFC !important;
        }
    
        /* Adjust the main content block if needed */
        .main .block-container {
            background-color: #F8FAFC;
        }
        /* Only removing padding, keeping header visible for Rerun button */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
        }

        /* Custom Metric Cards Design */
        [data-testid="stMetric"] {
            background-color: #ffffff;
            border: 1px solid #f0f2f6;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            transition: 0.3s;
        }
        [data-testid="stMetric"]:hover {
            border-color: #DAA520; /* Gold border on hover */
            box-shadow: 0 6px 8px rgba(0, 0, 0, 0.1);
        }
        /* Global Frame for Selectboxes */
        div[data-testid="stSelectbox"] [data-baseweb="select"] {
            border: 1px solid #dedede !important;
            border-radius: 10px !important;
            background-color: white !important;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.05) !important;
        }
        /* Custom card container for selection logic */
        .selection-card {
            border: 1px solid #dedede !important;
            border-radius: 15px !important;
            padding: 20px !important;
            background-color: #ffffff !important;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.05) !important;
            margin-bottom: 20px !important;
        }
        /* Global Frame for Radio Buttons Group */
        div[data-testid="stRadio"] > div[role="radiogroup"] {
            border: 1px solid #dedede !important;
            border-radius: 10px !important;
            padding: 15px !important;
            background-color: white !important;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.05) !important;
        }
        /* Styling for Radio labels (Optional) */
        div[data-testid="stRadio"] label p {
            font-weight: 600 !important;
            color: #444 !important;
        }
    </style>
    """, unsafe_allow_html=True)


local_css()

# Initialize session state for interactive elements
if 'selected_country' not in st.session_state:
    st.session_state.selected_country = 'Israel'

# --- 3. LOAD DATA (With Spinner to prevent white screen) ---
try:
    with st.spinner('Loading Olympic Data... Please wait...'):
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

    # --- 4. SIDEBAR NAVIGATION ---
    with st.sidebar:
        # Center the image using columns
        try:
            col1, col2, col3 = st.columns([0.5, 2, 0.5])
            with col2:
                st.image(icon_image, use_container_width=True)
        except:
            pass

        st.write("")  # Spacer

        # New Modern Navigation with Gold Icons
        selected_page = option_menu(
            menu_title="Navigation",
            options=["Global Overview", "Host Advantage", "Athletics Deep Dive", "Wellness & Winning"],
            # Changed back to 'person-running' as requested
            icons=['globe', 'house', 'trophy', 'graph-up-arrow'],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "5!important", "background-color": "#fafafa"},
                "icon": {"color": "#DAA520", "font-size": "20px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "#e6e6e6", "color": "black", "font-weight": "bold"},
            }
        )

    st.sidebar.divider()

    # --- 5. PAGE ROUTING ---
    if selected_page == "Global Overview":
        from global_overview import show_global_overview

        show_global_overview(medals_only, total_medals_per_country, country_list, medals_data)

    elif selected_page == "Host Advantage":
        from host_advantage import show_host_advantage

        show_host_advantage(host_data, medals_only, country_ref)

    elif selected_page == "Athletics Deep Dive":
        from athletics_deep_dive import show_athletics_deep_dive

        show_athletics_deep_dive(athletics_df, country_ref)

    elif selected_page == "Wellness & Winning":
        from wellness_winning import show_wellness_winning

        show_wellness_winning(gap_df)

except Exception as e:
    st.error(f"Error loading datasets: {e}")
