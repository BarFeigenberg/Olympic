import os
import numpy as np
import pandas as pd
import re
import streamlit as st
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

    # --- FIX: Plotly Choropleth Compatibility ---
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
    if countries.empty: return pd.DataFrame()

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
    if df.empty: return {}
    return dict(zip(df['noc'], df['country']))


def get_continent_mapping():
    df = load_raw_continent_data()
    if df.empty: return {}
    df.columns = df.columns.str.lower()
    return dict(zip(df['alpha-3'], df['region']))


@st.cache_data
def get_processed_athletics_data():
    df = load_raw_athletics_data()
    if df.empty: return pd.DataFrame()

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

    if df_medals.empty: return pd.DataFrame()

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
    """
    Merges historical population data with a specific 2024 update file.
    - Historical data is used for years < 2024.
    - New file is used EXCLUSIVELY for 2024.
    """
    # Load raw datasets
    hist_pop = load_historical_population_data()
    curr_pop = load_2024_population_data()

    # --- 1. Process Historical Data ---
    if hist_pop.empty:
        # If historical data is missing, create an empty structure
        combined_df = pd.DataFrame(columns=['country', 'year', 'population', 'iso'])
    else:
        # Normalize column names
        hist_pop.columns = hist_pop.columns.str.lower().str.strip()

        # Rename columns to standard format
        rename_dict = {}
        if 'entity' in hist_pop.columns: rename_dict['entity'] = 'country'
        if 'code' in hist_pop.columns: rename_dict['code'] = 'iso'

        # Identify population column in historical data
        pop_col = next((c for c in hist_pop.columns if 'population' in c), None)
        if pop_col: rename_dict[pop_col] = 'population'

        hist_pop = hist_pop.rename(columns=rename_dict)

        # Apply country name standardization
        if 'country' in hist_pop.columns:
            hist_pop['country'] = hist_pop['country'].replace(country_names_to_change)

        # Select only relevant columns
        combined_df = hist_pop[['country', 'year', 'population', 'iso']].copy()

        # Remove any existing 2024 data from history to prefer the new file
        combined_df = combined_df[combined_df['year'] != 2024]

    # --- 2. Process New 2024 Data ---
    if not curr_pop.empty:
        # Normalize columns
        curr_pop.columns = curr_pop.columns.str.lower().str.strip()

        # A. Identify 'Country' Column
        country_col = None
        possible_country_names = ['country', 'name', 'nation', 'entity']

        # Try finding exact name match
        for col in curr_pop.columns:
            if col in possible_country_names:
                country_col = col
                break

        # Fallback: Find first text-based column
        if not country_col:
            for col in curr_pop.columns:
                if curr_pop[col].dtype == object:
                    country_col = col
                    break

        # B. Identify 'Year' Column
        year_col = None
        if 'year' in curr_pop.columns:
            year_col = 'year'

        # C. Identify 'Population' Column
        # Look for 'population', 'pop', '2024', or take the first numeric column that isn't the year
        pop_col = None
        possible_pop_names = ['population', 'pop', 'total', '2024']

        # Try explicit match
        for col in curr_pop.columns:
            if any(x in col for x in possible_pop_names) and col != year_col:
                pop_col = col
                break

        # Fallback: Find first numeric column that is NOT the year
        if not pop_col:
            for col in curr_pop.columns:
                # Check if numeric and not the identified year column
                is_numeric = pd.api.types.is_numeric_dtype(curr_pop[col])
                if is_numeric and col != year_col:
                    pop_col = col
                    break

        # --- D. Clean and Format New Data ---
        if country_col and pop_col:
            # Create a clean subset
            new_data = pd.DataFrame()
            new_data['country'] = curr_pop[country_col]

            # Handle Year: If exists, use it. If not, force 2024.
            if year_col:
                new_data['year'] = pd.to_numeric(curr_pop[year_col], errors='coerce')
            else:
                new_data['year'] = 2024

            # Handle Population: Clean string formatting (remove commas)
            if curr_pop[pop_col].dtype == object:
                new_data['population'] = curr_pop[pop_col].astype(str).str.replace(',', '').apply(pd.to_numeric,
                                                                                                  errors='coerce')
            else:
                new_data['population'] = pd.to_numeric(curr_pop[pop_col], errors='coerce')

            # Filter ONLY for 2024 (as requested)
            new_data = new_data[new_data['year'] == 2024].copy()

            # Standardize country names
            new_data['country'] = new_data['country'].str.strip()
            new_data['country'] = new_data['country'].replace(country_names_to_change)

            # Attempt to map ISO codes from history
            if not combined_df.empty and 'iso' in combined_df.columns:
                iso_map = combined_df.dropna(subset=['iso']).set_index('country')['iso'].to_dict()
                new_data['iso'] = new_data['country'].map(iso_map)
            else:
                new_data['iso'] = np.nan

            # Append new 2024 data to the historical data
            combined_df = pd.concat([combined_df, new_data], ignore_index=True)

    # --- 3. Final Cleanup & Interpolation ---
    combined_df['year'] = combined_df['year'].astype(int)

    # Sort to ensure correct interpolation order
    combined_df = combined_df.sort_values(['country', 'year'])

    # Remove duplicates: keep the last entry (prioritizing the new file if overlap exists)
    combined_df = combined_df.drop_duplicates(subset=['country', 'year'], keep='last')

    # Interpolate missing years (fills gaps between history and 2024)
    min_year, max_year = 1896, 2024
    all_years = list(range(min_year, max_year + 1))
    olympic_years = list(range(min_year, max_year + 1, 4))

    processed_dfs = []

    for country, group in combined_df.groupby('country'):
        group = group.set_index('year')
        # Reindex to include all years for interpolation
        new_index = sorted(list(set(all_years) | set(group.index)))
        group = group.reindex(new_index)

        # Linear interpolation
        group['population'] = group['population'].interpolate(method='linear', limit_direction='both')
        group['country'] = country

        # Fill ISO forward/backward
        if 'iso' in group.columns:
            group['iso'] = group['iso'].ffill().bfill()

        # Filter only Olympic years
        olympic_data = group.loc[group.index.isin(olympic_years)].reset_index()
        processed_dfs.append(olympic_data)

    if not processed_dfs:
        return pd.DataFrame()

    final_df = pd.concat(processed_dfs, ignore_index=True)
    final_df = final_df.rename(columns={'index': 'year'})

    return final_df


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