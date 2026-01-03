# data_processor.py
# This file performs data cleaning, merging, and complex statistical calculations
import pandas as pd
import streamlit as st
import numpy as np
import re
from utils import get_name_map, NOC_TO_CONTINENT
from data_loader import load_raw_games_data, load_raw_tally_data, load_raw_main_data, load_raw_country_data, load_raw_athletics_data, load_gapminder_reference

def get_processed_host_data():
    # Calculates medal lifts for host countries compared to their historical averages
    try:
        return pd.read_csv('host_advantage_data.csv')
    except FileNotFoundError:
        pass

    games_df = load_raw_games_data()
    tally_df = load_raw_tally_data()

    if games_df.empty or tally_df.empty: return pd.DataFrame()

    games_summer = games_df[(games_df['edition'].str.contains('Summer', case=False, na=False)) & (games_df['year'] < 2024)].copy()
    tally_summer = tally_df[(tally_df['edition'].str.contains('Summer', case=False, na=False)) & (tally_df['year'] < 2024)].copy()

    hosts = games_summer[['year', 'city', 'country_noc']].rename(columns={'year': 'Year', 'city': 'Host_City', 'country_noc': 'Host_NOC'}).drop_duplicates(subset=['Year'])
    global_totals = tally_summer.groupby('year')['total'].sum().reset_index().rename(columns={'total': 'Global_Total', 'year': 'Year'})

    host_adv = pd.merge(hosts, tally_summer[['year', 'country_noc', 'total']], left_on=['Year', 'Host_NOC'], right_on=['year', 'country_noc'], how='left')
    host_adv.rename(columns={'total': 'Total_Medals'}, inplace=True)
    host_adv['Total_Medals'] = host_adv['Total_Medals'].fillna(0)

    host_adv = pd.merge(host_adv, global_totals, on='Year', how='left')
    host_adv = host_adv[host_adv['Global_Total'] > 0].copy()
    host_adv['Medal_Percentage'] = host_adv['Total_Medals'] / host_adv['Global_Total']

    all_pct = pd.merge(tally_summer, global_totals, left_on='year', right_on='Year', how='inner')
    all_pct['Calc_Percentage'] = all_pct['total'] / all_pct['Global_Total']

    avgs = []
    for _, row in host_adv.iterrows():
        hist = all_pct[(all_pct['country_noc'] == row['Host_NOC']) & (all_pct['year'] != row['Year'])]
        avgs.append(hist['Calc_Percentage'].mean() if not hist.empty else 0)

    host_adv['Avg_Percentage'] = avgs
    host_adv['Lift'] = host_adv.apply(lambda x: x['Medal_Percentage'] / x['Avg_Percentage'] if x['Avg_Percentage'] > 0 else 0, axis=1)

    host_adv.to_csv('host_advantage_data.csv', index=False)
    return host_adv
@st.cache_data
def get_processed_main_data():
    # Prepares the main medals dataset for global overview analysis
    df = load_raw_main_data()
    if df.empty: return pd.DataFrame()

    if 'year' in df.columns: df.rename(columns={'year': 'Year'}, inplace=True)
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df.dropna(subset=['Year'])
    df['Year'] = df['Year'].astype(int)

    # ==========================================
    # DATA CLEANING: Handle Historical Countries
    # ==========================================

    # 1. Merge historical German entities into modern Germany (GER)
    # GDR = East Germany, FRG = West Germany, EUA = United Team of Germany
    if 'NOC' in df.columns:
        df['NOC'] = df['NOC'].replace({
            'GDR': 'GER',
            'FRG': 'GER',
            'EUA': 'GER'
        })

        # 2. Remove dissolved empires/federations
        # URS = Soviet Union, TCH = Czechoslovakia, YUG = Yugoslavia
        # EUN = Unified Team (1992), BWI = British West Indies, BOH = Bohemia, ANZ = Australasia
        countries_to_remove = ['URS', 'TCH', 'YUG', 'EUN', 'BWI', 'BOH', 'ANZ', 'AHO']
        df = df[~df['NOC'].isin(countries_to_remove)]
    # ==========================================

    countries = load_raw_country_data()
    if not countries.empty:
        # Merge with country names
        df = df.merge(countries, left_on="NOC", right_on="noc", how="left")

    return df


@st.cache_data
def get_processed_athletics_data():
    from data_loader import load_raw_athletics_data
    from utils import get_name_map

    df = load_raw_athletics_data()
    if df.empty: return pd.DataFrame()

    df.drop(columns=['Extra'], inplace=True, errors='ignore')
    ref = get_name_map()
    df['Country'] = df['Nationality'].map(ref).fillna(df['Nationality'])

    # Event Cleaning
    df['BaseEvent'] = df['Event'].apply(
        lambda e: 'Sprint Hurdles' if 'Hurdles' in str(e) and ('110M' in str(e) or '100M' in str(e)) else str(
            e).replace(' Men', '').replace(' Women', '').strip())

    # --- ROBUST PARSING FUNCTION ---
    def parse_result(row):
        val = str(row['Result']).strip().lower()

        # Skip invalid strings
        if not val or val in ['dnf', 'dns', 'dq', 'nm', 'nan']:
            return np.nan

        # 1. Try direct float conversion (for points like 8462.235)
        try:
            return float(val)
        except ValueError:
            pass

        # 2. Handle Time Formats (HH:MM:SS or MM:SS.ms)
        try:
            # Standardize separators
            val = val.replace('h', ':').replace('-', ':').replace(';', ':')
            # Extract all numeric parts
            parts = [float(p) for p in re.findall(r'\d+\.?\d*', val)]

            if len(parts) == 3:  # Format: H:M:S (e.g., 0:27:06)
                return parts[0] * 3600 + parts[1] * 60 + parts[2]
            elif len(parts) == 2:  # Format: M:S (e.g., 27:06)
                # If it's a long distance event, treat first part as Hours if it makes sense
                if any(x in str(row['BaseEvent']) for x in ['Marathon', 'Walk']):
                    return parts[0] * 3600 + parts[1] * 60
                return parts[0] * 60 + parts[1]
            elif len(parts) == 1:
                return parts[0]
        except:
            return np.nan
        return np.nan

    # Apply the new logic
    df['NumericResult'] = df.apply(parse_result, axis=1)
    return df

@st.cache_data
def get_processed_gapminder_data():
    # Merges Olympic performance with Gapminder socioeconomic data
    df_main = load_raw_main_data()
    df_medals = load_raw_tally_data()
    if df_main.empty or df_medals.empty: return pd.DataFrame()

    df_medals = df_medals[df_medals['edition'].str.contains('Summer', na=False)].copy()
    df_medals = df_medals[['year', 'country_noc', 'total']].rename(columns={'total': 'Medals', 'country_noc': 'NOC', 'year': 'Year'})

    if 'year' in df_main.columns: df_main.rename(columns={'year': 'Year'}, inplace=True)
    delegation = df_main.groupby(['Year', 'NOC'])['player_id'].nunique().reset_index().rename(columns={'player_id': 'Delegation_Size'})

    stats = pd.merge(delegation, df_medals, on=['Year', 'NOC'], how='left').fillna({'Medals': 0})
    stats['Region'] = stats['NOC'].map(NOC_TO_CONTINENT).fillna('Western Europe')

    gapminder = load_gapminder_reference()
    years = range(1920, 2025)
    all_combos = pd.MultiIndex.from_product([gapminder['country'].unique(), years], names=['country', 'year']).to_frame(index=False)

    gap_full = pd.merge(all_combos, gapminder[['country', 'year', 'pop', 'gdpPercap', 'lifeExp', 'iso_alpha', 'continent']], on=['country', 'year'], how='left')

    for col in ['pop', 'gdpPercap', 'lifeExp']:
        gap_full[col] = gap_full.groupby('country')[col].transform(lambda x: x.interpolate(method='linear', limit_direction='both'))

    gap_full['iso_alpha'] = gap_full.groupby('country')['iso_alpha'].ffill().bfill()
    gap_full['continent'] = gap_full.groupby('country')['continent'].ffill().bfill()

    noc_iso = {'GER': 'DEU', 'NED': 'NLD', 'GRE': 'GRC', 'DEN': 'DNK', 'SUI': 'CHE', 'RSA': 'ZAF', 'GBR': 'GBR', 'CHN': 'CHN', 'USA': 'USA'}
    stats['ISO'] = stats['NOC'].map(noc_iso).fillna(stats['NOC'])

    df = pd.merge(stats, gap_full.rename(columns={'year': 'Year'}), left_on=['ISO', 'Year'], right_on=['iso_alpha', 'Year'], how='left')
    df['Population'] = df['pop'].fillna(5000000)
    df['Medals_Per_Million'] = (df['Medals'] / (df['Population'] / 1000000))
    df.rename(columns={'lifeExp': 'Life_Expectancy'}, inplace=True)

    ref = get_name_map()
    df['Country_Name'] = stats['NOC'].map(ref).fillna(stats['NOC'])

    return df[df['Medals'] > 0].sort_values(['Year', 'Country_Name'])