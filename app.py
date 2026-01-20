from streamlit_option_menu import option_menu
from data_processor import *
from data_loader import *
from PIL import Image

pd.set_option('future.no_silent_downcasting', True)

# Load app icon
icon_image = Image.open("olympics ring.png")

# --- 1. CONFIGURATION (Streamlit page setup) ---
st.set_page_config(
    layout="wide",
    page_title="Olympics Dashboard",
    page_icon=icon_image
)

# --- 2. CUSTOM CSS FOR BETTER UI ---
def local_css():
    """
    Inject custom CSS to enhance Streamlit UI:
    - Background color
    - Metric card styling with hover effect
    - Selectbox and Radio button styling
    - Custom card containers
    """
    st.markdown("""
    <style>
        /* Background for main app */
        .stApp { background-color: #F8FAFC; }
        header[data-testid="stHeader"] { background-color: #F8FAFC !important; }
        .main .block-container { background-color: #F8FAFC; }
        .block-container { padding-top: 1rem; padding-bottom: 2rem; }

        /* Metric card design */
        [data-testid="stMetric"] {
            background-color: #ffffff;
            border: 1px solid #f0f2f6;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            transition: 0.3s;
        }
        [data-testid="stMetric"]:hover {
            border-color: #DAA520;
            box-shadow: 0 6px 8px rgba(0,0,0,0.1);
        }

        /* Selectbox styling */
        div[data-testid="stSelectbox"] [data-baseweb="select"] {
            border: 1px solid #dedede !important;
            border-radius: 10px !important;
            background-color: white !important;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.05) !important;
        }

        /* Card container for selection */
        .selection-card {
            border: 1px solid #dedede !important;
            border-radius: 15px !important;
            padding: 20px !important;
            background-color: #ffffff !important;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.05) !important;
            margin-bottom: 20px !important;
        }

        /* Radio button styling */
        div[data-testid="stRadio"] > div[role="radiogroup"] {
            border: 1px solid #dedede !important;
            border-radius: 10px !important;
            padding: 15px !important;
            background-color: white !important;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.05) !important;
        }
        div[data-testid="stRadio"] label p {
            font-weight: 600 !important;
            color: #444 !important;
        }
    </style>
    """, unsafe_allow_html=True)


local_css()

# --- 3. INITIALIZE SESSION STATE ---
if 'selected_country' not in st.session_state:
    st.session_state.selected_country = 'Israel'  # default selected country

# --- 4. LOAD DATA WITH SPINNER ---
try:
    with st.spinner('Loading Olympic Data... Please wait...'):
        # Main datasets
        data = get_processed_main_data()
        host_data = create_host_advantage_file()
        athletics_df = get_processed_athletics_data()
        gap_df = get_processed_gapminder_data()
        country_ref = load_raw_country_data()
        medals_data = get_processed_total_medals_data()

        # Filter medals only and remove duplicates
        medals_only = data[data['medal'] != 'No medal'].drop_duplicates(subset=['year','event','noc','medal'])
        total_medals_per_country = medals_data.groupby('country')['total'].sum().reset_index()
        country_list = sorted(medals_only['team'].dropna().unique())

    # --- 5. SIDEBAR NAVIGATION ---
    with st.sidebar:
        # Center logo image
        try:
            col1, col2, col3 = st.columns([0.5,2,0.5])
            with col2:
                st.image(icon_image, width='stretch')
        except:
            pass

        st.write("")  # Spacer

        # Option menu for page navigation
        selected_page = option_menu(
            menu_title="Navigation",
            options=["Project Overview", "Global Overview", "Host Advantage", "Athletics Deep Dive", "Wellness & Winning"],
            icons=['info-circle', 'globe', 'house', 'trophy', 'graph-up-arrow'],
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

    # --- 6. PAGE ROUTING ---
    if selected_page == "Project Overview":
        from project_overview import show_project_overview
        show_project_overview()

    elif selected_page == "Global Overview":
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
    st.error(f"Error loading datasets: {e}")  # Display error if any dataset fails to load
