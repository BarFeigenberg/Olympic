import os
import numpy as np
import re
from data_loader import *

# ============================================================
# Country Name & NOC Standardization Dictionaries
# ============================================================
# These mappings normalize country names and NOC codes across
# different datasets, historical periods, and encoding issues.
# The goal is to ensure consistency for grouping, merging,
# and visualization (especially choropleth maps).
# ============================================================

country_names_to_change = {
    # Fixes for alternate official names, political entities,
    # historical states, and encoding/character issues
    "Chinese Taipei": "Taiwan",
    "C├┤te d'Ivoire": "Ivory Coast",
    "Democratic People's Republic of Korea": "North Korea",
    "Federated States of Micronesia": "Micronesia",
    "Hong Kong, China": "Hong Kong",
    "Independent Olympic Athletes": "Independent Athletes",
    "Individual Neutral Athletes": "Independent Athletes",
    "Islamic Republic of Iran": "Iran",
    "Kingdom of Saudi Arabia": "Saudi Arabia",
    "Lao People's Democratic Republic": "Laos",
    "Malaya": "Malaysia",
    "Newfoundland": "Canada",
    "North Borneo": "Malaysia",
    "North Yemen": "Yemen",
    "People's Republic of China": "China",
    "Refugee Olympic Team": "Independent Athletes",
    "Republic of Korea": "South Korea",
    "Republic of Moldova": "Moldova",
    "ROC": "Russia",
    "Russian Federation": "Russia",
    "Russian Olympic Committee": "Russia",
    "Saar": "Germany",
    "S╓│┬úo Tom╓│┬⌐ and Pr╓│┬¡ncipe": "Sao Tome and Principe",
    "S├úo Tom├⌐ and Pr├¡ncipe": "Sao Tome and Principe",
    "Serbia and Montenegro": "Serbia",
    "South Vietnam": "Vietnam",
    "South Yemen": "Yemen",
    "Syrian Arab Republic": "Syria",
    "T╓│┬╝rkiye": "Turkey",
    "T├╝rkiye": "Turkey",
    "United Arab Republic": "Egypt",
    "United Republic of Tanzania": "Tanzania",
    "United States Virgin Islands": "US Virgin Islands",
    "West Indies Federation": "West Indies",

    # Name alignment for Plotly choropleth compatibility
    "The Bahamas": "Bahamas",
    "Czechia": "Czech Republic",
    "Cabo Verde": "Cape Verde",
    "North Macedonia": "Macedonia",
    "Eswatini": "Swaziland",
    "Timor-Leste": "East Timor",
    "Brunei Darussalam": "Brunei",
    "Republic of the Congo": "Republic of Congo",
    "Democratic Republic of the Congo": "Dem. Rep. Congo",
    "Congo, Democratic Republic of the": "Dem. Rep. Congo",
    "Congo, Republic of the": "Republic of Congo",

    # Ensuring full, non-abbreviated country names
    "USSR": "Soviet Union",
    "UK": "United Kingdom",
    "Great Britain": "United Kingdom",
    "USA": "United States",
    "United States of America": "United States",
    "UAE": "United Arab Emirates",
    "United Arab Emirates": "United Arab Emirates",

    # Merging historical splits into modern entities
    "West Germany": "Germany",
    "East Germany": "Germany",
    "Yugoslavia": "Yugoslavia",
    "Czechoslovakia": "Czechoslovakia"
}

country_NOC_to_change = {
    # Normalization of historical and transitional NOC codes
    "AIN": "IOA",
    "MAL": "MAS",
    "NFL": "CAN",
    "NBO": "MAS",
    "YAR": "YAM",
    "EOR": "IOA",
    "ROC": "RUS",
    "SAA": "GER",
    "SCG": "SRB",
    "VNM": "VIE",
    "YMD": "YAM",
    "EUN": "URS",
    "UAR": "EGY",
    "FRG": "GER",
    "GDR": "GER"
}


# ============================================================
# Sport Category Mapping (Used for Aggregation & Visualization)
# ============================================================
# Maps raw sport names into a fixed set of balanced categories.
# This abstraction enables clearer comparisons across years
# and avoids fragmentation caused by sport name variations.
# ============================================================

SPORT_CATEGORIES = {
    # Athletics
    "Athletics": "🏃 Athletics",

    # Aquatics (pool + open water)
    "Swimming": "🏊 Aquatics", "Diving": "🏊 Aquatics", "Water Polo": "🏊 Aquatics",
    "Artistic Swimming": "🏊 Aquatics", "Synchronized Swimming": "🏊 Aquatics",
    "Marathon Swimming": "🏊 Aquatics", "Marathon Swimming, Swimming": "🏊 Aquatics",

    # Gymnastics disciplines
    "Gymnastics": "🤸 Gymnastics", "Artistic Gymnastics": "🤸 Gymnastics",
    "Rhythmic Gymnastics": "🤸 Gymnastics", "Trampoline Gymnastics": "🤸 Gymnastics",
    "Trampolining": "🤸 Gymnastics",

    # Combat sports
    "Wrestling": "🥋 Combat Sports", "Boxing": "🥋 Combat Sports", "Judo": "🥋 Combat Sports",
    "Fencing": "🥋 Combat Sports", "Taekwondo": "🥋 Combat Sports", "Karate": "🥋 Combat Sports",

    # Team ball sports
    "Basketball": "⚽ Ball Games", "3x3 Basketball": "⚽ Ball Games",
    "3x3 Basketball, Basketball": "⚽ Ball Games", "Volleyball": "⚽ Ball Games",
    "Beach Volleyball": "⚽ Ball Games", "Handball": "⚽ Ball Games",
    "Football": "⚽ Ball Games", "Hockey": "⚽ Ball Games", "Rugby": "⚽ Ball Games",
    "Rugby Sevens": "⚽ Ball Games", "Baseball": "⚽ Ball Games", "Softball": "⚽ Ball Games",
    "Baseball/Softball": "⚽ Ball Games", "Cricket": "⚽ Ball Games", "Lacrosse": "⚽ Ball Games",
    "Polo": "⚽ Ball Games", "Ice Hockey": "⚽ Ball Games",

    # Racquet-based sports
    "Tennis": "🏸 Racquet Sports", "Badminton": "🏸 Racquet Sports",
    "Table Tennis": "🏸 Racquet Sports", "Basque Pelota": "🏸 Racquet Sports",
    "Racquets": "🏸 Racquet Sports", "Jeu De Paume": "🏸 Racquet Sports",

    # Cycling disciplines
    "Cycling": "🚴 Cycling", "Cycling BMX Freestyle": "🚴 Cycling",
    "Cycling BMX Racing": "🚴 Cycling", "Cycling Mountain Bike": "🚴 Cycling",
    "Cycling Road": "🚴 Cycling", "Cycling Road, Cycling Mountain Bike": "🚴 Cycling",
    "Cycling Road, Cycling Track": "🚴 Cycling", "Cycling Road, Triathlon": "🚴 Cycling",
    "Cycling Track": "🚴 Cycling", "BMX": "🚴 Cycling",

    # Non-pool water sports
    "Rowing": "🚣 Water Sports", "Canoeing": "🚣 Water Sports",
    "Canoe Slalom": "🚣 Water Sports", "Canoe Sprint": "🚣 Water Sports",
    "Sailing": "🚣 Water Sports", "Surfing": "🚣 Water Sports", "Motorboating": "🚣 Water Sports",

    # Precision / target sports
    "Shooting": "🎯 Target Sports", "Archery": "🎯 Target Sports", "Golf": "🎯 Target Sports",

    # Strength-focused sports
    "Weightlifting": "🏋️ Strength & Weight", "Tug-Of-War": "🏋️ Strength & Weight",

    # Miscellaneous & modern additions
    "Triathlon": "🧗 Misc & Modern", "Modern Pentathlon": "🧗 Misc & Modern",
    "Equestrian": "🧗 Misc & Modern", "Equestrianism": "🧗 Misc & Modern",
    "Skateboarding": "🧗 Misc & Modern", "Sport Climbing": "🧗 Misc & Modern",
    "Breaking": "🧗 Misc & Modern", "Figure Skating": "🧗 Misc & Modern",
    "Art Competitions": "🧗 Misc & Modern", "Aeronautics": "🧗 Misc & Modern",
    "Alpinism": "🧗 Misc & Modern", "Croquet": "🧗 Misc & Modern", "Roque": "🧗 Misc & Modern"
}

# Explicit category order to ensure consistent axis ordering in charts
CATEGORY_ORDER = [
    "🏃 Athletics", "🏊 Aquatics", "🤸 Gymnastics", "🥋 Combat Sports",
    "⚽ Ball Games", "🏸 Racquet Sports", "🚴 Cycling", "🚣 Water Sports",
    "🎯 Target Sports", "🏋️ Strength & Weight", "🧗 Misc & Modern"
]


def get_sport_category(sport_name):
    """Maps a raw sport name to its high-level category."""
    if pd.isna(sport_name):
        return "Other"
    return SPORT_CATEGORIES.get(sport_name, "Other")


def get_medals_by_sport_category(medals_df, noc, year=None, weight_col=None):
    """
    Aggregates medal data by sport category for a given country.
    Optionally filters by year and supports weighted aggregation
    (e.g. scores instead of raw medal counts).
    """
    if medals_df.empty:
        return {cat: 0 for cat in CATEGORY_ORDER}

    # Filter medals by country (and year if provided)
    df = medals_df[medals_df['noc'] == noc].copy()
    if year is not None:
        df = df[df['year'] == year]

    if df.empty:
        return {cat: 0 for cat in CATEGORY_ORDER}

    # Assign each medal to a sport category
    df['category'] = df['sport'].apply(get_sport_category)

    # Aggregate either by sum (weighted) or by count
    if weight_col and weight_col in df.columns:
        counts = df.groupby('category')[weight_col].sum().to_dict()
    else:
        counts = df.groupby('category').size().to_dict()

    # Ensure all categories are present in the output
    return {cat: counts.get(cat, 0) for cat in CATEGORY_ORDER}


# ============================================================
# Host Advantage Computation
# ============================================================
# Measures how much host countries overperform (or underperform)
# relative to their historical medal share.
# ============================================================

def create_host_advantage_file():
    # Load raw datasets
    games_df = load_raw_games_data()
    tally_df = load_raw_tally_data()
    paris_df = load_raw_paris_data()

    # Early exit if critical data is missing
    if games_df.empty or tally_df.empty:
        return pd.DataFrame()

    # Normalize column names for consistency
    games_df.columns = games_df.columns.str.lower()
    tally_df.columns = tally_df.columns.str.lower()

    games_df['country_noc'] = games_df['country_noc'].replace(country_NOC_to_change)
    tally_df['country_noc'] = tally_df['country_noc'].replace(country_NOC_to_change)

    # Restrict analysis to Summer Olympics prior to 2024
    games_summer = games_df[
        (games_df['edition'].str.contains('Summer', case=False, na=False)) & (games_df['year'] < 2024)].copy()
    tally_summer = tally_df[
        (tally_df['edition'].str.contains('Summer', case=False, na=False)) & (tally_df['year'] < 2024)].copy()
    tally_summer = tally_summer.groupby(['year', 'country_noc'])['total'].sum().reset_index()
    # Extract host country per year
    hosts = games_summer[['year', 'city', 'country_noc']].rename(
        columns={'city': 'host_city', 'country_noc': 'host_noc'})
    hosts = hosts.drop_duplicates(subset=['year'])

    # Compute global medal totals per year
    global_totals = tally_summer.groupby('year')['total'].sum().reset_index()
    global_totals.rename(columns={'total': 'global_total'}, inplace=True)

    # Merge host medal counts with host metadata
    host_advantage_df = pd.merge(
        hosts,
        tally_summer[['year', 'country_noc', 'total']],
        left_on=['year', 'host_noc'],
        right_on=['year', 'country_noc'],
        how='left'
    )

    host_advantage_df = host_advantage_df[['year', 'host_city', 'host_noc', 'total']].rename(
        columns={'total': 'total_medals'})
    host_advantage_df['total_medals'] = host_advantage_df['total_medals'].fillna(0)

    # Attach global totals and compute medal share
    host_advantage_df = pd.merge(host_advantage_df, global_totals, on='year', how='left')
    host_advantage_df['medal_percentage'] = (
        host_advantage_df['total_medals'] / host_advantage_df['global_total']
    )

    # Compute historical average medal share for each country
    all_Percentage = pd.merge(tally_summer, global_totals, on='year', how='inner')
    all_Percentage['calc_percentage'] = (
        all_Percentage['total'] / all_Percentage['global_total']
    )

    avg_Percentage = []
    for _, row in host_advantage_df.iterrows():
        country_history = all_Percentage[
            (all_Percentage['country_noc'] == row['host_noc']) &
            (all_Percentage['year'] != row['year'])
        ]
        avg_Percentage.append(
            country_history['calc_percentage'].mean() if not country_history.empty else 0
        )

    host_advantage_df['avg_percentage'] = avg_Percentage

    # Host advantage defined as deviation from historical average
    host_advantage_df['lift'] = host_advantage_df['medal_percentage'] - host_advantage_df['avg_percentage']

    # Special handling for Paris 2024 (not included in historical datasets)
    if not paris_df.empty:
        paris_df.columns = paris_df.columns.str.lower()
        france_row = paris_df[paris_df['country_code'] == 'FRA']

        if not france_row.empty:
            total_medals_france = france_row['total'].values[0]
            total_medals_global_2024 = paris_df['total'].sum()

            france_history = all_Percentage[all_Percentage['country_noc'] == 'FRA']
            avg_france = (
                france_history['calc_percentage'].mean()
                if not france_history.empty else 0
            )

            curr_pct = total_medals_france / total_medals_global_2024

            paris_row = pd.DataFrame([{
                'year': 2024,
                'host_city': 'Paris',
                'host_noc': 'FRA',
                'total_medals': total_medals_france,
                'global_total': total_medals_global_2024,
                'medal_percentage': curr_pct,
                'avg_percentage': avg_france,
                'lift': curr_pct - avg_france
            }])

            host_advantage_df = pd.concat(
                [host_advantage_df, paris_row],
                ignore_index=True
            )

    return host_advantage_df.sort_values('year')



def calculate_host_advantage_from_tally(metric_type='Total Medals'):
    """
    Calculates host advantage stats using the PRE-PROCESSED TALLY data.
    This reuses the robust cleaning logic from `get_processed_total_medals_data`
    and avoids duplication while allowing dynamic metric selection.
    """
    # 1. Load CLEANED and COMBINED data (Historical + Paris)
    tally_data = get_processed_total_medals_data()
    games_df = load_raw_games_data()

    if tally_data.empty or games_df.empty:
        return pd.DataFrame()

    # 2. Prepare Host Data (Summer only)
    games_df.columns = games_df.columns.str.lower()
    games_df['country_noc'] = games_df['country_noc'].replace(country_NOC_to_change)
    games_summer = games_df[
        (games_df['edition'].str.contains('Summer', case=False, na=False)) & 
        (games_df['year'].between(1896, 2024))
    ].copy()
    
    # Extract host country per year
    hosts = games_summer[['year', 'city', 'country_noc']].rename(
        columns={'city': 'host_city', 'country_noc': 'host_noc'})
    hosts = hosts.drop_duplicates(subset=['year'])

    # 3. Filter Tally Data for Summer Games (just in case, though get_processed usually does this)
    tally_summer = tally_data[tally_data['edition'].str.contains('Summer', case=False, na=False)].copy()

    # 4. Determine Metric Column
    # Ensure numeric columns
    for c in ['gold', 'silver', 'bronze', 'total']:
        if c in tally_summer.columns:
            tally_summer[c] = pd.to_numeric(tally_summer[c], errors='coerce').fillna(0)

    target_col = 'total' # default
    if metric_type == 'Gold Medals':
        target_col = 'gold'
    elif metric_type == 'Weighted Score':
        tally_summer['score'] = (tally_summer['gold'] * 3) + (tally_summer['silver'] * 2) + (tally_summer['bronze'] * 1)
        target_col = 'score'

    # 5. Group by Year/NOC
    tally_grouped = tally_summer.groupby(['year', 'country_noc'])[target_col].sum().reset_index()
    tally_grouped.rename(columns={target_col: 'metric_value'}, inplace=True)

    # 6. Compute Global Totals
    global_totals = tally_grouped.groupby('year')['metric_value'].sum().reset_index()
    global_totals.rename(columns={'metric_value': 'global_total'}, inplace=True)

    # 7. Merge Host Info
    host_advantage_df = pd.merge(
        hosts,
        tally_grouped,
        left_on=['year', 'host_noc'],
        right_on=['year', 'country_noc'],
        how='left'
    )

    host_advantage_df = host_advantage_df[['year', 'host_city', 'host_noc', 'metric_value']].rename(
        columns={'metric_value': 'total_medals'})
    host_advantage_df['total_medals'] = host_advantage_df['total_medals'].fillna(0)

    # 8. Compute Shares
    host_advantage_df = pd.merge(host_advantage_df, global_totals, on='year', how='left')
    host_advantage_df['medal_percentage'] = (
        host_advantage_df['total_medals'] / host_advantage_df['global_total']
    )

    # 9. Compute Historical Average (Lift Logic)
    all_Percentage = pd.merge(tally_grouped, global_totals, on='year', how='inner')
    all_Percentage['calc_percentage'] = (
        all_Percentage['metric_value'] / all_Percentage['global_total']
    )

    avg_Percentage = []
    for _, row in host_advantage_df.iterrows():
        country_history = all_Percentage[
            (all_Percentage['country_noc'] == row['host_noc']) &
            (all_Percentage['year'] != row['year'])
        ]
        avg_Percentage.append(
            country_history['calc_percentage'].mean() if not country_history.empty else 0
        )

    host_advantage_df['avg_percentage'] = avg_Percentage
    host_advantage_df['lift'] = host_advantage_df['medal_percentage'] - host_advantage_df['avg_percentage']

    return host_advantage_df.sort_values('year')



# ============================================================
# Base Data Processors
# ============================================================
# Functions for cleaning and preparing core Olympics datasets.
# Includes country normalization, main results processing,
# athletics-specific records, and numeric result parsing.
# ============================================================

@st.cache_data
def get_processed_country_data():
    """
    Load and clean country-level data.
    - Applies name and NOC standardizations
    - Removes duplicates
    - Saves a cleaned copy for reuse
    """
    countries = load_raw_country_data()
    if countries.empty:
        return pd.DataFrame()

    countries.columns = countries.columns.str.lower()

    # Standardize country names and NOC codes
    countries['country'] = countries['country'].replace(country_names_to_change)
    countries['noc'] = countries['noc'].replace(country_NOC_to_change)

    # Remove duplicate NOCs
    countries = countries.drop_duplicates(subset=['noc'])

    # Save cleaned file
    countries.to_csv('Olympics_Country_Cleaned.csv', index=False)
    if os.path.exists('Olympics_Country_Cleaned.csv'):
        os.replace('Olympics_Country_Cleaned.csv', 'Olympics_Country.csv')

    return countries


@st.cache_data
def get_processed_main_data():
    """
    Load and clean main Olympics dataset.
    - Converts 'year' to numeric
    - Applies NOC corrections
    - Merges country info
    - Standardizes medal labels
    """
    df = load_raw_main_data()
    if df.empty:
        return pd.DataFrame()

    df.columns = df.columns.str.lower()

    # Ensure 'year' is numeric and drop invalid rows
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['year'])
    df['year'] = df['year'].astype(int)

    # Get cleaned country reference
    countries = get_processed_country_data()

    # Fix known NOC inconsistencies
    df.loc[df['noc'] == 'LIB', 'noc'] = 'LBN'
    df.loc[df['noc'] == 'ROT', 'noc'] = 'IOA'
    df['noc'] = df['noc'].replace(country_NOC_to_change)

    # Merge country info
    merged_df = df.merge(countries, left_on='noc', right_on='noc', how='left')

    # Fill 'team' column using country name if missing
    if 'team' in merged_df.columns:
        merged_df['team'] = merged_df['country'].fillna(merged_df['team'])
    else:
        merged_df['team'] = merged_df['country']

    df = merged_df.drop(columns=['country'], errors='ignore')

    # Clean and standardize 'medal' column
    if 'medal' in df.columns:
        df['medal'] = df['medal'].astype(str).str.strip().str.title()
        medal_map = {'G': 'Gold', 'S': 'Silver', 'B': 'Bronze'}
        df['medal'] = df['medal'].replace(medal_map)

    return df


def get_name_map():
    """
    Returns a dictionary mapping NOC codes to standardized country names.
    Used for mapping nationality codes in athletics or other datasets.
    """
    df = get_processed_country_data()
    if df.empty:
        return {}
    return dict(zip(df['noc'], df['country']))


@st.cache_data
def get_all_world_countries():
    """
    Return list of all world countries from continent_data.csv.
    - Applies name standardizations
    """
    df = load_raw_continent_data()
    if df.empty:
        return []

    df.columns = df.columns.str.lower()
    countries = df['name'].dropna().unique().tolist()

    # Apply standardization mapping
    standardized = [country_names_to_change.get(c, c) for c in countries]

    return list(set(standardized))


def get_continent_mapping():
    """
    Return dictionary mapping alpha-3 codes to continent/region names.
    """
    df = load_raw_continent_data()
    if df.empty:
        return {}
    df.columns = df.columns.str.lower()
    return dict(zip(df['alpha-3'], df['region']))


@st.cache_data
def get_processed_athletics_data():
    """
    Load athletics data, merge with historical records, and process results.
    - Adds manually curated meet_records for important events
    - Deduplicates by gender, event, year, result
    - Maps nationality to country
    - Creates simplified 'baseevent' for grouping
    - Converts results to numeric values in seconds/meters
    """
    df = load_raw_athletics_data()

    # Manually curated historic meet records
    meet_records = [
        # Example entries: Gold medal performances, male and female
        {'gender': 'M', 'event': '100M', 'location': 'London', 'year': 2012, 'medal': 'G', 'name': 'Usain Bolt',
         'nationality': 'JAM', 'result': '9.63'},
        # ... (many more records omitted for brevity in comment)
    ]

    records_df = pd.DataFrame(meet_records)

    if df.empty:
        df = records_df
    else:
        df.columns = df.columns.str.lower()
        df = pd.concat([df, records_df], ignore_index=True)

    # Deduplicate by key columns
    df = df.drop_duplicates(subset=['gender', 'event', 'year', 'result'], keep='last')

    # Drop irrelevant columns if present
    df.drop(columns=['extra'], inplace=True, errors='ignore')

    # Map nationality to standardized country
    ref = get_name_map()
    if 'nationality' in df.columns:
        df['country'] = df['nationality'].map(ref).fillna(df['nationality'])

    # Simplify event names for grouping
    if 'event' in df.columns:
        df['baseevent'] = df['event'].apply(
            lambda e: 'Sprint Hurdles' if 'Hurdles' in str(e) and ('110M' in str(e) or '100M' in str(e))
            else str(e).replace(' Men', '').replace(' Women', '').strip()
        )

    # ============================================================
    # Parse result strings into numeric values
    # Handles:
    # - standard float values
    # - times in H:M:S or M:S format
    # - invalid or missing entries ('DNF', 'DNS', 'DQ', 'NM', 'nan')
    # ============================================================
    def parse_result(row):
        val = str(row.get('result', '')).strip().lower()
        if not val or val in ['dnf', 'dns', 'dq', 'nm', 'nan']:
            return np.nan
        try:
            return float(val)
        except ValueError:
            pass
        try:
            val = val.replace('h', ':').replace('-', ':').replace(';', ':')
            parts = [float(p) for p in re.findall(r'\d+\.?\d*', val)]
            if len(parts) == 3:
                return parts[0] * 3600 + parts[1] * 60 + parts[2]
            elif len(parts) == 2:
                base_ev = str(row.get('baseevent', ''))
                if any(x in base_ev for x in ['Marathon', 'Walk']):
                    return parts[0] * 3600 + parts[1] * 60
                return parts[0] * 60 + parts[1]
            elif len(parts) == 1:
                return parts[0]
        except:
            return np.nan
        return np.nan

    df['numericresult'] = df.apply(parse_result, axis=1)
    return df


@st.cache_data
def get_processed_total_medals_data():
    df_medals = load_raw_tally_data()
    df_paris_medals = load_raw_paris_data()
    countries = get_processed_country_data()

    if df_medals.empty:
        return pd.DataFrame()

    df_medals.columns = df_medals.columns.str.lower()
    if not df_paris_medals.empty:
        df_paris_medals.columns = df_paris_medals.columns.str.lower()

    df_medals = df_medals[df_medals['edition'].str.contains('Summer', na=False)].copy()
    df_medals['country_noc'] = df_medals['country_noc'].replace(country_NOC_to_change)
    merged_df = df_medals.merge(countries, left_on='country_noc', right_on='noc', how='left')

    if 'country_y' in merged_df.columns:
        merged_df['country_x'] = merged_df['country_y'].fillna(merged_df.get('country_x', merged_df['country_noc']))
        df_medals = merged_df.drop(columns=['country_y', 'noc', 'edition_id'], errors='ignore')

    df_medals = df_medals.rename(columns={'country_x': 'country'})

    if not df_paris_medals.empty and "2024 Summer Olympics" not in df_medals['edition'].values:
        rename_map = {
            'country_code': 'country_noc',
            'gold medal': 'gold',
            'silver medal': 'silver',
            'bronze medal': 'bronze',
            'total': 'total'
        }
        df_paris_medals.rename(columns=rename_map, inplace=True)

        df_paris_medals['edition'] = '2024 Summer Olympics'
        df_paris_medals['year'] = 2024

        df_paris_medals['country_noc'] = df_paris_medals['country_noc'].replace(country_NOC_to_change)

        merged_df = df_paris_medals.merge(countries, left_on='country_noc', right_on='noc', how='left')
        if 'country_y' in merged_df.columns:
            merged_df['country_x'] = merged_df['country_y'].fillna(merged_df.get('country_x', merged_df['country_noc']))

        df_paris_medals = merged_df.rename(columns={'country_x': 'country'})

        columns_order = ['edition', 'year', 'country', 'country_noc', 'gold', 'silver', 'bronze', 'total']
        cols_to_keep = [c for c in columns_order if c in df_paris_medals.columns]
        df_paris_medals = df_paris_medals[cols_to_keep]
        updated_medals_df = pd.concat([df_medals, df_paris_medals], ignore_index=True)
    else:
        updated_medals_df = df_medals

    updated_medals_df.to_csv('Olympic_Games_Medal_Tally_Updated.csv', index=False)
    if os.path.exists('Olympic_Games_Medal_Tally_Updated.csv'):
        os.replace('Olympic_Games_Medal_Tally_Updated.csv', 'Olympic_Games_Medal_Tally.csv')

    return updated_medals_df


@st.cache_data
def get_processed_medals_data_by_score_and_type():
    """
    Returns a DataFrame with medal counts by type (Gold, Silver, Bronze)
    and a weighted score (G=3, S=2, B=1).
    Deduplicates events so that team sports count as 1 medal per country per event.
    Returns DETAILED event-level data (preserving Sport/Event columns) for historical data.
    """
    # --- 1. Historical Data (Detailed) ---
    df = get_processed_main_data()
    
    if not df.empty:
        # Filter for actual medals
        df = df[df['medal'].isin(['Gold', 'Silver', 'Bronze'])].copy()
        
        # Deduplicate: One medal per Event per Country per Year
        # This is the CRITICAL fix: We count Events, not Athletes.
        df_dedup = df.drop_duplicates(subset=['year', 'event', 'noc', 'medal']).copy()
        
        # Add basic metrics
        df_dedup['total_count'] = 1
        
        weights = {'Gold': 3, 'Silver': 2, 'Bronze': 1}
        df_dedup['score'] = df_dedup['medal'].map(weights).fillna(0)
        
        # Create dummy columns for consistency with previous schema (if needed by other consumers)
        # But primarily we want the detailed rows. 
        # For compatibility with groupby consumers, mapped 1/0 columns are useful.
        df_dedup['Gold'] = (df_dedup['medal'] == 'Gold').astype(int)
        df_dedup['Silver'] = (df_dedup['medal'] == 'Silver').astype(int)
        df_dedup['Bronze'] = (df_dedup['medal'] == 'Bronze').astype(int)
        
        historical_detailed = df_dedup
    else:
        historical_detailed = pd.DataFrame(columns=['team', 'noc', 'year', 'sport', 'event', 'medal', 'Gold', 'Silver', 'Bronze', 'score', 'total_count'])

    # --- 2. Paris 2024 Data (Tally - No Sport Detail) ---
    df_paris = load_raw_paris_data()
    if not df_paris.empty:
        # Clean columns
        df_paris.columns = df_paris.columns.str.lower().str.strip()
        
        # Rename strictly to match our schema (Total counts only)
        rename_map = {
            'gold medal': 'Gold',
            'silver medal': 'Silver',
            'bronze medal': 'Bronze',
            'country_code': 'noc',
            'country': 'team'
        }
        df_paris = df_paris.rename(columns=rename_map)
        df_paris['year'] = 2024
        
        # CLEANUP: Clean NOCs
        df_paris['noc'] = df_paris['noc'].replace(country_NOC_to_change)
        
         # CLEANUP: Get Standardized Team Names using our central country processor
        countries_ref = get_processed_country_data()
        if not countries_ref.empty:
             df_paris = df_paris.merge(countries_ref[['noc', 'country']], on='noc', how='left')
             df_paris.rename(columns={'country': 'team_std'}, inplace=True)
             df_paris['team'] = df_paris['team_std'].fillna(df_paris.get('team', df_paris['noc']))
             df_paris.drop(columns=['team_std'], inplace=True, errors='ignore')

        # Ensure numeric
        for c in ['Gold', 'Silver', 'Bronze']:
             df_paris[c] = pd.to_numeric(df_paris.get(c, 0), errors='coerce').fillna(0).astype(int)

        # Calculate Scores
        df_paris['score'] = (df_paris['Gold'] * 3) + (df_paris['Silver'] * 2) + (df_paris['Bronze'] * 1)
        df_paris['total_count'] = df_paris['Gold'] + df_paris['Silver'] + df_paris['Bronze']
        
        # Create "Fake" detailed rows? 
        # No, we just append. Columns 'sport', 'event', 'medal' will be NaN.
        # This implies Paris data won't show up in Breakdown charts (because no sport),
        # but will work for Totals/KPIs (summing columns).
        paris_final = df_paris
    else:
        paris_final = pd.DataFrame()

    # --- 3. Combine ---
    # Concatenate using compatible columns.
    # We prioritize Historical Detailed because it has Sports.
    # If 2024 is in historical, we ignore Paris Tally (as detail is better).
    if 2024 in historical_detailed['year'].values:
        final_df = historical_detailed
    else:
        final_df = pd.concat([historical_detailed, paris_final], ignore_index=True)
        
    return final_df


@st.cache_data
def calculate_host_advantage_stats(metric_col='total_count'):
    """
    Compute Host Advantage statistics dynamically for a given metric.
    - metric_col: column to calculate (e.g., 'total_count', 'score', 'Gold')
    - Calculates current host year performance vs historical average
    - Computes 'lift' as ratio of host year share to historical share
    """
    # 1. Load base medal data
    source_df = get_processed_medals_data_by_score_and_type()
    if source_df.empty:
        return pd.DataFrame()

    # 2. Load host information
    games_df = load_raw_games_data()  # Year -> Host city/NOC mapping
    if games_df.empty:
        return pd.DataFrame()
    games_df.columns = games_df.columns.str.lower()

    # Filter for Summer Games
    games_summer = games_df[games_df['edition'].str.contains('Summer', case=False, na=False)][['year', 'city', 'country_noc']]

    # Ensure Paris 2024 is included
    if 2024 not in games_summer['year'].values:
        paris_row = pd.DataFrame([{'year': 2024, 'city': 'Paris', 'country_noc': 'FRA'}])
        games_summer = pd.concat([games_summer, paris_row], ignore_index=True)

    hosts_map = games_summer.drop_duplicates(subset=['year']).set_index('year')

    # 3. Compute global totals per year for the selected metric
    global_totals = source_df.groupby('year')[metric_col].sum().reset_index()
    global_totals.rename(columns={metric_col: 'global_total'}, inplace=True)

    results = []
    known_years = sorted(hosts_map.index.unique())

    # Precompute share for all rows for efficiency
    df_with_global = source_df.merge(global_totals, on='year', how='left')
    df_with_global['share'] = df_with_global[metric_col] / df_with_global['global_total']

    for year in known_years:
        host_info = hosts_map.loc[year]
        if isinstance(host_info, pd.DataFrame):
            host_info = host_info.iloc[0]  # handle duplicates

        h_noc = host_info['country_noc']
        h_city = host_info['city']

        # Current host year performance
        host_perf = df_with_global[(df_with_global['year'] == year) & (df_with_global['noc'] == h_noc)]
        if host_perf.empty:
            continue

        current_total = host_perf[metric_col].sum()
        current_share = host_perf['share'].sum()
        global_total = host_perf['global_total'].values[0]

        # Historical average for this country excluding host year
        history = df_with_global[(df_with_global['noc'] == h_noc) & (df_with_global['year'] != year)]
        avg_share = history['share'].mean() if not history.empty else 0

        # Compute lift
        lift = (current_share / avg_share) if avg_share > 0 else 0

        results.append({
            'year': year,
            'host_city': h_city,
            'host_noc': h_noc,
            'total_medals': current_total,  # metric value
            'global_total': global_total,
            'medal_percentage': current_share,
            'avg_percentage': avg_share,
            'lift': lift
        })

    final_df = pd.DataFrame(results)
    if not final_df.empty:
        final_df = final_df.sort_values('year')

    return final_df


# --- 4. Population Processor ---
@st.cache_data
def get_combined_population_data():
    """
    Combine historical and 2024 population datasets.
    - Historical data for < 2024
    - New 2024 file for 2024
    - Interpolates missing years
    - Standardizes country names and ISO codes
    """
    # Load datasets
    hist_pop = load_historical_population_data()
    curr_pop = load_2024_population_data()

    # 1. Process historical data
    if hist_pop.empty:
        combined_df = pd.DataFrame(columns=['country', 'year', 'population', 'iso'])
    else:
        hist_pop.columns = hist_pop.columns.str.lower().str.strip()
        rename_dict = {}
        if 'entity' in hist_pop.columns: rename_dict['entity'] = 'country'
        if 'code' in hist_pop.columns: rename_dict['code'] = 'iso'

        pop_col = next((c for c in hist_pop.columns if 'population' in c), None)
        if pop_col: rename_dict[pop_col] = 'population'

        hist_pop = hist_pop.rename(columns=rename_dict)

        # Standardize country names
        if 'country' in hist_pop.columns:
            hist_pop['country'] = hist_pop['country'].replace(country_names_to_change)

        combined_df = hist_pop[['country', 'year', 'population', 'iso']].copy()
        combined_df = combined_df[combined_df['year'] != 2024]  # exclude 2024 to prioritize new file

    # 2. Process new 2024 population data
    if not curr_pop.empty:
        curr_pop.columns = curr_pop.columns.str.lower().str.strip()

        # Identify key columns
        country_col = next((col for col in curr_pop.columns if col in ['country','name','nation','entity']), None)
        if not country_col:
            country_col = next((col for col in curr_pop.columns if curr_pop[col].dtype == object), None)

        year_col = 'year' if 'year' in curr_pop.columns else None

        pop_col = next((col for col in curr_pop.columns
                        if any(x in col for x in ['population','pop','total','2024']) and col != year_col), None)
        if not pop_col:
            pop_col = next((col for col in curr_pop.columns
                            if pd.api.types.is_numeric_dtype(curr_pop[col]) and col != year_col), None)

        if country_col and pop_col:
            new_data = pd.DataFrame()
            new_data['country'] = curr_pop[country_col]
            new_data['year'] = pd.to_numeric(curr_pop[year_col], errors='coerce') if year_col else 2024
            new_data['population'] = pd.to_numeric(curr_pop[pop_col].astype(str).str.replace(',', ''), errors='coerce') \
                if curr_pop[pop_col].dtype == object else pd.to_numeric(curr_pop[pop_col], errors='coerce')

            # Handle 'Millions' scale
            scale_col = next((c for c in curr_pop.columns if 'scale' in c.lower()), None)
            if scale_col:
                is_millions = curr_pop[scale_col].astype(str).str.lower().str.contains('million', na=False)
                new_data.loc[is_millions, 'population'] *= 1_000_000

            # Keep only 2024
            new_data = new_data[new_data['year'] == 2024].copy()

            # Standardize country names
            new_data['country'] = new_data['country'].str.strip().replace(country_names_to_change)

            # Map ISO codes from historical data
            if not combined_df.empty and 'iso' in combined_df.columns:
                iso_map = combined_df.dropna(subset=['iso']).set_index('country')['iso'].to_dict()
                new_data['iso'] = new_data['country'].map(iso_map)
            else:
                new_data['iso'] = np.nan

            combined_df = pd.concat([combined_df, new_data], ignore_index=True)

    # 3. Final cleanup & interpolation
    combined_df['year'] = combined_df['year'].astype(int)
    combined_df = combined_df.sort_values(['country', 'year'])
    combined_df = combined_df.drop_duplicates(subset=['country', 'year'], keep='last')

    # Interpolate missing years for Olympic data
    min_year, max_year = 1896, 2024
    all_years = list(range(min_year, max_year+1))
    olympic_years = list(range(min_year, max_year+1, 4))
    processed_dfs = []

    for country, group in combined_df.groupby('country'):
        group = group.set_index('year')
        new_index = sorted(list(set(all_years) | set(group.index)))
        group = group.reindex(new_index)
        group['population'] = group['population'].interpolate(method='linear', limit_direction='both')
        group['country'] = country
        if 'iso' in group.columns:
            group['iso'] = group['iso'].ffill().bfill()
        olympic_data = group.loc[group.index.isin(olympic_years)].reset_index()
        processed_dfs.append(olympic_data)

    if not processed_dfs:
        return pd.DataFrame()

    final_df = pd.concat(processed_dfs, ignore_index=True).rename(columns={'index': 'year'})
    return final_df


# --- 5. Life Expectancy Processor (Wide Format Fix) ---
@st.cache_data
def get_processed_life_expectancy_data():
    """
    Process life expectancy data:
    - Converts wide format (years as columns) to long format
    - Cleans and standardizes country names
    - Interpolates missing values to ensure all Olympic years are covered
    """
    raw_lex = load_life_expectancy_data()
    if raw_lex.empty:
        return pd.DataFrame()

    # 1. Normalize column names
    raw_lex.columns = raw_lex.columns.str.lower().str.strip()

    # 2. Rename main columns
    if 'name' in raw_lex.columns:
        raw_lex = raw_lex.rename(columns={'name': 'country'})

    # 3. Detect wide format and melt if necessary
    year_cols = [col for col in raw_lex.columns if col.isdigit()]
    if year_cols:
        id_vars = ['country']
        if 'geo' in raw_lex.columns:
            id_vars.append('geo')
        raw_lex = raw_lex.melt(id_vars=id_vars, value_vars=year_cols, var_name='year', value_name='life_expectancy')
    else:
        # Handle long format fallback
        rename_map = {col: 'year' if 'year' in col or 'time' in col else 'life_expectancy'
                      for col in raw_lex.columns if 'life' in col or 'lex' in col}
        raw_lex.rename(columns=rename_map, inplace=True)

    # 4. Standard cleaning
    if 'country' not in raw_lex.columns or 'year' not in raw_lex.columns or 'life_expectancy' not in raw_lex.columns:
        return pd.DataFrame()

    raw_lex['country'] = raw_lex['country'].astype(str).str.strip().replace(country_names_to_change)
    raw_lex['year'] = pd.to_numeric(raw_lex['year'], errors='coerce')
    raw_lex['life_expectancy'] = pd.to_numeric(raw_lex['life_expectancy'], errors='coerce')
    raw_lex = raw_lex.dropna(subset=['year', 'country', 'life_expectancy'])
    raw_lex['year'] = raw_lex['year'].astype(int)

    # 5. Interpolate for all Olympic years
    min_year, max_year = 1896, 2024
    all_years = list(range(min_year, max_year + 1))
    olympic_years = list(range(min_year, max_year + 1, 4))
    processed_dfs = []

    for country, group in raw_lex.groupby('country'):
        group = group.set_index('year')
        new_index = sorted(list(set(all_years) | set(group.index)))
        group = group.reindex(new_index)
        group['life_expectancy'] = group['life_expectancy'].interpolate(method='linear', limit_direction='both')
        group['country'] = country
        olympic_data = group.loc[group.index.isin(olympic_years)].reset_index()
        processed_dfs.append(olympic_data)

    if not processed_dfs:
        return pd.DataFrame()

    final_lex = pd.concat(processed_dfs, ignore_index=True).rename(columns={'index': 'year'})
    return final_lex


# --- 6. Main Gapminder Processor ---
def calculate_medals_per_million(df):
    """
    Compute medals per million population.
    - Handles NaN population by filling with 1M to avoid division by zero
    """
    if df is None or df.empty:
        return df
    if 'medals' not in df.columns or 'population' not in df.columns:
        return df

    if df['population'].isnull().any():
        df['population'] = df['population'].fillna(1_000_000)

    df['medals_per_million'] = df['medals'] / (df['population'] / 1_000_000)
    return df


@st.cache_data
def get_processed_gapminder_data():
    """
    Combine main, medals, population, and life expectancy data:
    - Computes delegation size per NOC/year
    - Aggregates medals and scores
    - Merges population and life expectancy
    - Computes medals per million
    - Maps continents, applies manual overrides for historical/special NOCs
    - Filters only rows with medals
    """
    df_main = get_processed_main_data()
    df_medals_detailed = get_processed_medals_data_by_score_and_type()
    pop_df = get_combined_population_data()
    lex_df = get_processed_life_expectancy_data()

    if df_main.empty or df_medals_detailed.empty:
        return pd.DataFrame()

    # Calculate delegation size
    delegation = df_main.groupby(['year', 'noc'])['player_id'].nunique().reset_index().rename(
        columns={'player_id': 'delegation_size'})

    # Aggregate medals and score per NOC/year
    medals_agg = df_medals_detailed.groupby(['year', 'noc'])[['total_count', 'score']].sum().reset_index()
    medals_agg = medals_agg.rename(columns={'total_count': 'medals'})

    # Merge delegation size and medals
    stats = pd.merge(delegation, medals_agg, on=['year', 'noc'], how='left').fillna({'medals': 0, 'score': 0})

    # Map country names
    ref = get_name_map()
    stats['country'] = stats['noc'].map(ref).fillna(stats['noc']).astype(str).str.strip()

    # Merge population
    if not pop_df.empty:
        pop_df['country'] = pop_df['country'].astype(str).str.strip()
        df = pd.merge(stats, pop_df, on=['country', 'year'], how='left')
    else:
        df = stats.copy()
        df['population'] = np.nan
        df['iso'] = np.nan

    # Merge life expectancy
    if not lex_df.empty:
        lex_df['country'] = lex_df['country'].astype(str).str.strip()
        df = pd.merge(df, lex_df[['country', 'year', 'life_expectancy']], on=['country', 'year'], how='left')
    else:
        df['life_expectancy'] = np.nan

    # Compute medals per million
    df = calculate_medals_per_million(df)

    # Fill missing life expectancy with default
    df['life_expectancy'] = df['life_expectancy'].fillna(70)

    # Map continents using ISO or NOC, apply overrides
    noc_to_continent = get_continent_mapping()
    df['continent'] = df['iso'].map(noc_to_continent) if 'iso' in df.columns else np.nan
    df['continent'] = df['continent'].fillna(df['noc'].map(noc_to_continent))
    df['continent'] = df['continent'].fillna('Unknown')

    continent_overrides = {
        'YUG': 'Europe', 'TCH': 'Europe', 'URS': 'Europe', 'EUN': 'Europe',
        'GDR': 'Europe', 'FRG': 'Europe', 'EUA': 'Europe', 'BOH': 'Europe',
        'AHO': 'Americas', 'ANZ': 'Oceania', 'BWI': 'Americas', 'ROC': 'Europe',
        'SCG': 'Europe', 'RU1': 'Europe', 'SRB': 'Europe', 'MNE': 'Europe',
        'KOS': 'Europe'
    }
    mask_override = df['noc'].isin(continent_overrides)
    df.loc[mask_override, 'continent'] = df.loc[mask_override, 'noc'].map(continent_overrides)
    df['continent'] = df['continent'].fillna('Unknown')

    # Filter rows with medals only
    df = df[df['medals'] > 0].sort_values(['year', 'country'])

    # Alias for plotting
    df['country_name'] = df['country']

    # Normalize column names and remove duplicates
    df.columns = df.columns.str.lower()
    df = df.loc[:, ~df.columns.duplicated()]

    return df
