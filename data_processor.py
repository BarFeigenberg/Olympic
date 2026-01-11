import os
import numpy as np
import re
from data_loader import *

# --- 1. Dictionaries for Name Standardization ---
country_names_to_change = {
    # Standardization fixes
    "Chinese Taipei": "Taiwan",
    "Côte d'Ivoire": "Ivory Coast",
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
    "Sֳ£o Tomֳ© and Prֳ­ncipe": "Sao Tome and Principe",
    "São Tomé and Príncipe": "Sao Tome and Principe",
    "Serbia and Montenegro": "Serbia",
    "South Vietnam": "Vietnam",
    "South Yemen": "Yemen",
    "Syrian Arab Republic": "Syria",
    "Tֳ¼rkiye": "Turkey",
    "Türkiye": "Turkey",
    "United Arab Republic": "Egypt",
    "United Republic of Tanzania": "Tanzania",
    "United States Virgin Islands": "US Virgin Islands",
    "West Indies Federation": "West Indies",

    # Plotly Choropleth Compatibility ---
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

    # --- FIX: Ensuring Full Names (No Abbreviations) ---
    "USSR": "Soviet Union",
    "UK": "United Kingdom",
    "Great Britain": "United Kingdom",
    "USA": "United States",
    "United States of America": "United States",
    "UAE": "United Arab Emirates",
    "United Arab Emirates": "United Arab Emirates",

    # Merging historical entities
    "West Germany": "Germany",
    "East Germany": "Germany",
    "Yugoslavia": "Yugoslavia",
    "Czechoslovakia": "Czechoslovakia"
}

country_NOC_to_change = {
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

SPORT_CATEGORIES = {
    # 🏃 Athletics
    "Athletics": "Athletics 🏃",

    # 🏊 Aquatics
    "Swimming": "Aquatics 🏊", "Diving": "Aquatics 🏊", "Water Polo": "Aquatics 🏊",
    "Artistic Swimming": "Aquatics 🏊", "Synchronized Swimming": "Aquatics 🏊",
    "Marathon Swimming": "Aquatics 🏊", "Marathon Swimming, Swimming": "Aquatics 🏊",

    # 🤸 Gymnastics
    "Gymnastics": "Gymnastics 🤸", "Artistic Gymnastics": "Gymnastics 🤸",
    "Rhythmic Gymnastics": "Gymnastics 🤸", "Trampoline Gymnastics": "Gymnastics 🤸",
    "Trampolining": "Gymnastics 🤸",

    # 🥋 Combat Sports
    "Wrestling": "Combat Sports 🥋", "Boxing": "Combat Sports 🥋", "Judo": "Combat Sports 🥋",
    "Fencing": "Combat Sports 🥋", "Taekwondo": "Combat Sports 🥋", "Karate": "Combat Sports 🥋",

    # ⚽ Ball Games
    "Basketball": "Ball Games ⚽", "3x3 Basketball": "Ball Games ⚽",
    "3x3 Basketball, Basketball": "Ball Games ⚽", "Volleyball": "Ball Games ⚽",
    "Beach Volleyball": "Ball Games ⚽", "Handball": "Ball Games ⚽",
    "Football": "Ball Games ⚽", "Hockey": "Ball Games ⚽", "Rugby": "Ball Games ⚽",
    "Rugby Sevens": "Ball Games ⚽", "Baseball": "Ball Games ⚽", "Softball": "Ball Games ⚽",
    "Baseball/Softball": "Ball Games ⚽", "Cricket": "Ball Games ⚽", "Lacrosse": "Ball Games ⚽",
    "Polo": "Ball Games ⚽", "Ice Hockey": "Ball Games ⚽",

    # 🏸 Racquet Sports
    "Tennis": "Racquet Sports 🏸", "Badminton": "Racquet Sports 🏸",
    "Table Tennis": "Racquet Sports 🏸", "Basque Pelota": "Racquet Sports 🏸",
    "Racquets": "Racquet Sports 🏸", "Jeu De Paume": "Racquet Sports 🏸",

    # 🚴 Cycling
    "Cycling": "Cycling 🚴", "Cycling BMX Freestyle": "Cycling 🚴",
    "Cycling BMX Racing": "Cycling 🚴", "Cycling Mountain Bike": "Cycling 🚴",
    "Cycling Road": "Cycling 🚴", "Cycling Road, Cycling Mountain Bike": "Cycling 🚴",
    "Cycling Road, Cycling Track": "Cycling 🚴", "Cycling Road, Triathlon": "Cycling 🚴",
    "Cycling Track": "Cycling 🚴", "BMX": "Cycling 🚴",

    # 🚣 Water Sports
    "Rowing": "Water Sports 🚣", "Canoeing": "Water Sports 🚣",
    "Canoe Slalom": "Water Sports 🚣", "Canoe Sprint": "Water Sports 🚣",
    "Sailing": "Water Sports 🚣", "Surfing": "Water Sports 🚣", "Motorboating": "Water Sports 🚣",

    # 🎯 Target Sports
    "Shooting": "Target Sports 🎯", "Archery": "Target Sports 🎯", "Golf": "Target Sports 🎯",

    # 🏋️ Strength & Weight
    "Weightlifting": "Strength & Weight 🏋️", "Tug-Of-War": "Strength & Weight 🏋️",

    # 🧗 General Sport
    "Triathlon": "General Sport 🧗", "Modern Pentathlon": "General Sport 🧗",
    "Equestrian": "General Sport 🧗", "Equestrianism": "General Sport 🧗",
    "Skateboarding": "General Sport 🧗", "Sport Climbing": "General Sport 🧗",
    "Breaking": "General Sport 🧗", "Figure Skating": "General Sport 🧗",
    "Art Competitions": "General Sport 🧗", "Aeronautics": "General Sport 🧗",
    "Alpinism": "General Sport 🧗", "Croquet": "General Sport 🧗", "Roque": "General Sport 🧗"
}

CATEGORY_ORDER = [
    "Athletics 🏃", "Aquatics 🏊", "Gymnastics 🤸", "Combat Sports 🥋",
    "Ball Games ⚽", "Racquet Sports🏸", "Cycling 🚴", "Water Sports 🚣",
    "Target Sports 🎯", "️Strength & Weight 🏋", "General Sport 🧗"
]


def get_sport_category(sport_name):
    """Get category for a sport name"""
    if pd.isna(sport_name):
        return "Other"
    return SPORT_CATEGORIES.get(sport_name, "Other")


def get_medals_by_sport_category(medals_df, noc, year=None):
    """Aggregate medals by sport category for a country. Uses pre-deduplicated medals_df."""
    if medals_df.empty:
        return {cat: 0 for cat in CATEGORY_ORDER}
    
    df = medals_df[medals_df['noc'] == noc].copy()
    if year is not None:
        df = df[df['year'] == year]
    
    if df.empty:
        return {cat: 0 for cat in CATEGORY_ORDER}
    
    df['category'] = df['sport'].apply(get_sport_category)
    counts = df.groupby('category').size().to_dict()
    
    return {cat: counts.get(cat, 0) for cat in CATEGORY_ORDER}


# --- 2. Host Advantage Processor ---
def create_host_advantage_file():
    games_df = load_raw_games_data()
    tally_df = load_raw_tally_data()
    paris_df = load_raw_paris_data()

    if games_df.empty or tally_df.empty:
        return pd.DataFrame()

    # Enforce lowercase
    games_df.columns = games_df.columns.str.lower()
    tally_df.columns = tally_df.columns.str.lower()

    games_summer = games_df[
        (games_df['edition'].str.contains('Summer', case=False, na=False)) &
        (games_df['year'] < 2024)].copy()

    tally_summer = tally_df[
        (tally_df['edition'].str.contains('Summer', case=False, na=False)) &
        (tally_df['year'] < 2024)].copy()

    hosts = games_summer[['year', 'city', 'country_noc']].rename(columns={
        'city': 'host_city',
        'country_noc': 'host_noc'
    })

    hosts = hosts.drop_duplicates(subset=['year'])

    global_totals = tally_summer.groupby('year')['total'].sum().reset_index()
    global_totals.rename(columns={'total': 'global_total'}, inplace=True)

    host_advantage_df = pd.merge(
        hosts,
        tally_summer[['year', 'country_noc', 'total']],
        left_on=['year', 'host_noc'],
        right_on=['year', 'country_noc'],
        how='left'
    )

    host_advantage_df = host_advantage_df[['year', 'host_city', 'host_noc', 'total']]
    host_advantage_df.rename(columns={'total': 'total_medals'}, inplace=True)
    host_advantage_df['total_medals'] = host_advantage_df['total_medals'].fillna(0)

    host_advantage_df = pd.merge(host_advantage_df, global_totals, on='year', how='left')
    host_advantage_df = host_advantage_df[host_advantage_df['global_total'] > 0].copy()

    host_advantage_df['medal_percentage'] = host_advantage_df['total_medals'] / host_advantage_df['global_total']

    all_Percentage = pd.merge(tally_summer, global_totals, on='year', how='inner')
    all_Percentage['calc_percentage'] = all_Percentage['total'] / all_Percentage['global_total']

    avg_Percentage = []

    for index, row in host_advantage_df.iterrows():
        host_noc = row['host_noc']
        host_year = row['year']
        country_history = all_Percentage[
            (all_Percentage['country_noc'] == host_noc) &
            (all_Percentage['year'] != host_year)
            ]

        if not country_history.empty:
            avg = country_history['calc_percentage'].mean()
        else:
            avg = 0

        avg_Percentage.append(avg)

    host_advantage_df['avg_percentage'] = avg_Percentage
    host_advantage_df['lift'] = host_advantage_df.apply(
        lambda x: x['medal_percentage'] / x['avg_percentage'] if x['avg_percentage'] > 0 else 0, axis=1
    )

    if not paris_df.empty:
        paris_df.columns = paris_df.columns.str.lower()
        if 'total' not in paris_df.columns and 'Total' in paris_df.columns:
            paris_df.rename(columns={'Total': 'total'}, inplace=True)

        france_row = paris_df[paris_df['country_code'] == 'FRA']
        if not france_row.empty:
            total_medals_france = france_row['total'].values[0]
            total_medals_global_2024 = paris_df['total'].sum()
            france_history = all_Percentage[all_Percentage['country_noc'] == 'FRA']
            avg_france = france_history['calc_percentage'].mean() if not france_history.empty else 0
            current_percentage = total_medals_france / total_medals_global_2024
            lift_score = current_percentage / avg_france if avg_france > 0 else 0

            paris_row = pd.DataFrame([{
                'year': 2024,
                'host_city': 'Paris',
                'host_noc': 'FRA',
                'total_medals': total_medals_france,
                'global_total': total_medals_global_2024,
                'medal_percentage': current_percentage,
                'avg_percentage': avg_france,
                'lift': lift_score
            }])

            host_advantage_df = pd.concat([host_advantage_df, paris_row], ignore_index=True)

    host_advantage_df = host_advantage_df.sort_values('year')
    host_advantage_df.to_csv('host_advantage_data.csv', index=False)
    return host_advantage_df


# --- 3. Base Data Processors ---
@st.cache_data
def get_processed_country_data():
    countries = load_raw_country_data()
    if countries.empty:
        return pd.DataFrame()

    countries.columns = countries.columns.str.lower()

    # Apply standardizations
    countries['country'] = countries['country'].replace(country_names_to_change)
    countries['noc'] = countries['noc'].replace(country_NOC_to_change)
    countries = countries.drop_duplicates(subset=['noc'])

    countries.to_csv('Olympics_Country_Cleaned.csv', index=False)
    if os.path.exists('Olympics_Country_Cleaned.csv'):
        os.replace('Olympics_Country_Cleaned.csv', 'Olympics_Country.csv')
    return countries


@st.cache_data
def get_processed_main_data():
    df = load_raw_main_data()
    if df.empty: return pd.DataFrame()

    df.columns = df.columns.str.lower()

    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['year'])
    df['year'] = df['year'].astype(int)

    countries = get_processed_country_data()

    df.loc[df['noc'] == 'LIB', 'noc'] = 'LBN'
    df.loc[df['noc'] == 'ROT', 'noc'] = 'IOA'
    df['noc'] = df['noc'].replace(country_NOC_to_change)

    merged_df = df.merge(countries, left_on='noc', right_on='noc', how='left')

    if 'team' in merged_df.columns:
        merged_df['team'] = merged_df['country'].fillna(merged_df['team'])
    else:
        merged_df['team'] = merged_df['country']

    df = merged_df.drop(columns=['country'], errors='ignore')

    return df


def get_name_map():
    df = get_processed_country_data()
    if df.empty:
        return {}
    return dict(zip(df['noc'], df['country']))


@st.cache_data
def get_all_world_countries():
    """Get all world country names from continent_data.csv with standardized names"""
    df = load_raw_continent_data()
    if df.empty:
        return []

    df.columns = df.columns.str.lower()
    countries = df['name'].dropna().unique().tolist()

    # Apply our name standardizations
    standardized = [country_names_to_change.get(c, c) for c in countries]

    return list(set(standardized))


def get_continent_mapping():
    df = load_raw_continent_data()
    if df.empty: return {}
    df.columns = df.columns.str.lower()
    return dict(zip(df['alpha-3'], df['region']))


@st.cache_data
def get_processed_athletics_data():
    df = load_raw_athletics_data()
    # Normalize columns to lowercase immediately to match 'result', 'event', 'nationality' keys used below
    df.columns = df.columns.str.lower()

    df.drop(columns=['extra'], inplace=True, errors='ignore')
    ref = get_name_map()

    if 'nationality' in df.columns:
        df['country'] = df['nationality'].map(ref).fillna(df['nationality'])

    if 'event' in df.columns:
        df['baseevent'] = df['event'].apply(
            lambda e: 'Sprint Hurdles' if 'Hurdles' in str(e) and ('110M' in str(e) or '100M' in str(e)) else str(
                e).replace(' Men', '').replace(' Women', '').strip())

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
def get_processed_medals_data():
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


# --- 4. Population Processor ---
@st.cache_data
def get_combined_population_data():
    # 1. Load Historical Data
    hist_df = load_historical_population_data()
    if not hist_df.empty:
        # Expected cols: Entity, Code, Year, Population (historical)
        # Rename to standard
        hist_df = hist_df.rename(columns={
            'Entity': 'country',
            'Year': 'year',
            'Population (historical)': 'population'
        })
        hist_df = hist_df[['country', 'year', 'population']]

    # 2. Load 2024 Data
    df_2024 = load_2024_population_data()
    processed_2024 = []

    if not df_2024.empty:
        # Check standard columns from WEO file
        # 'Country', 'Scale', '2024'
        if 'Country' in df_2024.columns and '2024' in df_2024.columns:
            # Create a copy to match standard structure
            temp = df_2024[['Country', 'Scale', '2024']].copy()
            temp = temp.rename(columns={'Country': 'country'})

            # Handle Scaling (Millions)
            # Ensure '2024' is numeric
            temp['2024'] = pd.to_numeric(temp['2024'], errors='coerce')

            def scale_pop(row):
                val = row['2024']
                scale = str(row['Scale']).lower()
                if pd.isna(val):
                    return 0
                if 'million' in scale:
                    return val * 1_000_000
                return val

            temp['population'] = temp.apply(scale_pop, axis=1)
            temp['year'] = 2024

            processed_2024 = temp[['country', 'year', 'population']]

    # 3. Combine
    if hist_df.empty and isinstance(processed_2024, list):
        return pd.DataFrame()

    if isinstance(processed_2024, pd.DataFrame) and not processed_2024.empty:
        combined = pd.concat([hist_df, processed_2024], ignore_index=True)
    else:
        combined = hist_df

    return combined


# --- 5. Life Expectancy Processor (Wide Format Fix) ---
@st.cache_data
def get_processed_life_expectancy_data():
    raw_lex = load_life_expectancy_data()
    if raw_lex.empty:
        return pd.DataFrame()

    # 1. Inspect and Clean Column Names
    raw_lex.columns = raw_lex.columns.str.lower().str.strip()

    # 2. Rename the specific columns we found in lex.csv
    # The file has 'geo', 'name', and then years '1800'...'2100'
    if 'name' in raw_lex.columns:
        raw_lex = raw_lex.rename(columns={'name': 'country'})

    # 3. Check format: Wide (Years as columns) or Long?
    # If columns contain '1900', '2000' etc., it is wide.
    year_cols = [col for col in raw_lex.columns if col.isdigit()]

    if len(year_cols) > 0:
        # --- Handle Wide Format (Melting) ---
        # Keep 'country' (and 'geo' if useful, but country is key)
        id_vars = ['country']
        if 'geo' in raw_lex.columns:
            id_vars.append('geo')

        # Melt to Long Format
        raw_lex = raw_lex.melt(id_vars=id_vars, value_vars=year_cols, var_name='year', value_name='life_expectancy')
    else:
        # Fallback for Long Format logic (if needed in future)
        rename_map = {}
        for col in raw_lex.columns:
            if 'year' in col or 'time' in col:
                rename_map[col] = 'year'
            elif 'life' in col or 'lex' in col:
                rename_map[col] = 'life_expectancy'
        raw_lex.rename(columns=rename_map, inplace=True)

    # 4. Standard Cleaning
    if 'country' not in raw_lex.columns or 'year' not in raw_lex.columns or 'life_expectancy' not in raw_lex.columns:
        return pd.DataFrame()

    raw_lex['country'] = raw_lex['country'].astype(str).str.strip()
    raw_lex['country'] = raw_lex['country'].replace(country_names_to_change)

    raw_lex['year'] = pd.to_numeric(raw_lex['year'], errors='coerce')
    raw_lex['life_expectancy'] = pd.to_numeric(raw_lex['life_expectancy'], errors='coerce')

    raw_lex = raw_lex.dropna(subset=['year', 'country', 'life_expectancy'])
    raw_lex['year'] = raw_lex['year'].astype(int)

    # 5. Interpolation (To ensure Olympic years coverage)
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

    final_lex = pd.concat(processed_dfs, ignore_index=True)
    final_lex = final_lex.rename(columns={'index': 'year'})

    return final_lex


# --- 6. Main Gapminder Processor ---

def calculate_medals_per_million(df):
    """
    Calculates medals per million population for a given DataFrame.
    Requires 'medals' and 'population' columns.
    Handles NaN population by filling with 1M (to avoid division by zero/NaN).
    """
    if df is None or df.empty:
        return df

    # We need both columns to perform calculation
    if 'medals' not in df.columns or 'population' not in df.columns:
        return df

    # Fill NaN population with 1M (Default behavior extracted from original code)
    if df['population'].isnull().any():
        df['population'] = df['population'].fillna(1_000_000)

    df['medals_per_million'] = (df['medals'] / (df['population'] / 1_000_000))

    return df


@st.cache_data
def get_combined_population_data():
    # 1. Load Historical Data
    hist_df = load_historical_population_data()
    if not hist_df.empty:
        # Expected cols: Entity, Code, Year, Population (historical)
        # Rename to standard
        hist_df = hist_df.rename(columns={
            'Entity': 'country',
            'Year': 'year',
            'Population (historical)': 'population'
        })
        hist_df = hist_df[['country', 'year', 'population']]
    
    # 2. Load 2024 Data
    df_2024 = load_2024_population_data()
    processed_2024 = []
    
    if not df_2024.empty:
        # Check standard columns from WEO file
        # 'Country', 'Scale', '2024'
        if 'Country' in df_2024.columns and '2024' in df_2024.columns:
            # Create a copy to match standard structure
            temp = df_2024[['Country', 'Scale', '2024']].copy()
            temp = temp.rename(columns={'Country': 'country'})
            
            # Handle Scaling (Millions)
            # Ensure '2024' is numeric
            temp['2024'] = pd.to_numeric(temp['2024'], errors='coerce')
            
            def scale_pop(row):
                val = row['2024']
                scale = str(row['Scale']).lower()
                if pd.isna(val):
                    return 0
                if 'million' in scale:
                    return val * 1_000_000
                return val
            
            temp['population'] = temp.apply(scale_pop, axis=1)
            temp['year'] = 2024
            
            processed_2024 = temp[['country', 'year', 'population']]
    
    # 3. Combine
    if hist_df.empty and isinstance(processed_2024, list):
         return pd.DataFrame()
    
    if isinstance(processed_2024, pd.DataFrame) and not processed_2024.empty:
        combined = pd.concat([hist_df, processed_2024], ignore_index=True)
    else:
        combined = hist_df
        
    return combined


@st.cache_data
def get_processed_gapminder_data():
    df_main = get_processed_main_data()
    df_medals = get_processed_medals_data()
    pop_df = get_combined_population_data()
    lex_df = get_processed_life_expectancy_data()

    if df_main.empty or df_medals.empty:
        return pd.DataFrame()

    delegation = df_main.groupby(['year', 'noc'])['player_id'].nunique().reset_index().rename(
        columns={'player_id': 'delegation_size'})

    medals_subset = df_medals[['year', 'country_noc', 'total']].rename(columns={
        'country_noc': 'noc',
        'total': 'medals'
    })

    stats = pd.merge(delegation, medals_subset, on=['year', 'noc'], how='left').fillna({'medals': 0})

    ref = get_name_map()
    stats['country'] = stats['noc'].map(ref).fillna(stats['noc'])
    stats['country'] = stats['country'].astype(str).str.strip()

    if not pop_df.empty:
        pop_df['country'] = pop_df['country'].astype(str).str.strip()
        df = pd.merge(stats, pop_df, on=['country', 'year'], how='left')
    else:
        df = stats.copy()
        df['population'] = np.nan
        df['iso'] = np.nan

    if not lex_df.empty:
        lex_df['country'] = lex_df['country'].astype(str).str.strip()
        df = pd.merge(df, lex_df[['country', 'year', 'life_expectancy']], on=['country', 'year'], how='left')
    else:
        df['life_expectancy'] = np.nan

    # Use the new helper function for calculation
    df = calculate_medals_per_million(df)

    # If merge failed, this will set to 70.
    df['life_expectancy'] = df['life_expectancy'].fillna(70)

    noc_to_continent = get_continent_mapping()
    if 'iso' in df.columns:
        df['continent'] = df['iso'].map(noc_to_continent).fillna('Unknown')
    else:
        df['continent'] = 'Unknown'

    continent_overrides = {
        'YUG': 'Europe', 'TCH': 'Europe', 'URS': 'Europe', 'EUN': 'Europe',
        'GDR': 'Europe', 'FRG': 'Europe', 'EUA': 'Europe', 'BOH': 'Europe',
        'AHO': 'Americas', 'ANZ': 'Oceania', 'BWI': 'Americas', 'ROC': 'Europe',
        'SCG': 'Europe', 'RU1': 'Europe', 'SRB': 'Europe', 'MNE': 'Europe',
        'KOS': 'Europe'
    }
    df.loc[df['continent'] == 'Unknown', 'continent'] = df['noc'].map(continent_overrides)
    df['continent'] = df['continent'].fillna('Unknown')

    df = df[df['medals'] > 0].sort_values(['year', 'country'])

    # Required alias for the plotting library
    df['country_name'] = df['country']

    df.columns = df.columns.str.lower()
    df = df.loc[:, ~df.columns.duplicated()]

    return df
