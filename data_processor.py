import os
import numpy as np
import re
from data_loader import *


def create_host_advantage_file():
    games_df = load_raw_games_data()
    tally_df = load_raw_tally_data()
    paris_df = load_raw_paris_data()

    if games_df.empty or tally_df.empty:
        return pd.DataFrame()

    games_summer = games_df[
        (games_df['edition'].str.contains('Summer', case=False, na=False)) &
        (games_df['year'] < 2024)].copy()

    tally_summer = tally_df[
        (tally_df['edition'].str.contains('Summer', case=False, na=False)) &
        (tally_df['year'] < 2024)].copy()

    hosts = games_summer[['year', 'city', 'country_noc']].rename(columns={
        'year': 'Year',
        'city': 'Host_City',
        'country_noc': 'Host_NOC'
    })

    hosts = hosts.drop_duplicates(subset=['Year'])

    global_totals = tally_summer.groupby('year')['total'].sum().reset_index()
    global_totals.rename(columns={'total': 'Global_Total', 'year': 'Year'}, inplace=True)

    host_advantage_df = pd.merge(
        hosts,
        tally_summer[['year', 'country_noc', 'total']],
        left_on=['Year', 'Host_NOC'],
        right_on=['year', 'country_noc'],
        how='left'
    )

    host_advantage_df = host_advantage_df[['Year', 'Host_City', 'Host_NOC', 'total']]
    host_advantage_df.rename(columns={'total': 'Total_Medals'}, inplace=True)
    host_advantage_df['Total_Medals'] = host_advantage_df['Total_Medals'].fillna(0)

    host_advantage_df = pd.merge(host_advantage_df, global_totals, on='Year', how='left')
    host_advantage_df = host_advantage_df[host_advantage_df['Global_Total'] > 0].copy()

    host_advantage_df['Medal_Percentage'] = host_advantage_df['Total_Medals'] / host_advantage_df['Global_Total']

    all_Percentage = pd.merge(tally_summer, global_totals, left_on='year', right_on='Year', how='inner')
    all_Percentage['Calc_Percentage'] = all_Percentage['total'] / all_Percentage['Global_Total']

    avg_Percentage = []

    for index, row in host_advantage_df.iterrows():
        host_noc = row['Host_NOC']
        host_year = row['Year']
        country_history = all_Percentage[
            (all_Percentage['country_noc'] == host_noc) &
            (all_Percentage['year'] != host_year)
            ]

        if not country_history.empty:
            avg = country_history['Calc_Percentage'].mean()
        else:
            avg = 0

        avg_Percentage.append(avg)

    host_advantage_df['Avg_Percentage'] = avg_Percentage

    host_advantage_df['Lift'] = host_advantage_df.apply(
        lambda x: x['Medal_Percentage'] / x['Avg_Percentage'] if x['Avg_Percentage'] > 0 else 0, axis=1
    )

    france_row = paris_df[paris_df['country_code'] == 'FRA']
    total_medals_france = france_row['Total'].values[0]
    total_medals_global_2024 = paris_df['Total'].sum()
    france_history = all_Percentage[all_Percentage['country_noc'] == 'FRA']
    avg_france = france_history['Calc_Percentage'].mean() if not france_history.empty else 0
    current_percentage = total_medals_france / total_medals_global_2024
    lift_score = current_percentage / avg_france if avg_france > 0 else 0

    paris_row = pd.DataFrame([{
        'Year': 2024,
        'Host_City': 'Paris',
        'Host_NOC': 'FRA',
        'Total_Medals': total_medals_france,
        'Global_Total': total_medals_global_2024,
        'Medal_Percentage': current_percentage,
        'Avg_Percentage': avg_france,
        'Lift': lift_score
    }])

    host_advantage_df = pd.concat([host_advantage_df, paris_row], ignore_index=True)
    host_advantage_df = host_advantage_df.sort_values('Year')

    # 9. Save Output
    output_filename = 'host_advantage_data.csv'
    host_advantage_df.to_csv(output_filename, index=False)
    return host_advantage_df


country_names_to_change = {
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
    "Saar": "Germany",
    "Sֳ£o Tomֳ© and Prֳ­ncipe": "São Tomé and Príncipe",
    "Serbia and Montenegro": "Serbia",
    "South Vietnam": "Vietnam",
    "South Yemen": "Yemen",
    "Syrian Arab Republic": "Syria",
    "Tֳ¼rkiye": "Turkey",
    "United Arab Republic": "Egypt",
    "United Republic of Tanzania": "Tanzania",
    "United States Virgin Islands": "US Virgin Islands",
    "West Germany": "Germany",
    "West Indies Federation": "West Indies",
    "Russian Olympic Committee": "Russia"
}

country_NOC_to_change = {
    "AIN": "IOA",  # Individual Neutral Athletes to Independent Olympic Athletes
    "MAL": "MAS",  # Malaya to Malaysia
    "NFL": "CAN",  # Newfoundland to Canada
    "NBO": "MAS",  # North Borneo to Malaysia
    "YAR": "YAM",  # North Yemen to Yemen
    "EOR": "IOA",  # Refugee Olympic Team to Independent Olympic Athletes
    "ROC": "RUS",  # ROC and Russian Olympic Committee ro Russia
    "SAA": "GER",  # Saar to Germany
    "SCG": "SRB",  # Serbia and Montenegro to Serbia
    "VNM": "VIE",  # South Vietnam to Vietnam
    "YMD": "YAM",  # South Yemen to Yemen
    "EUN": "URS",  # Unified Team to Soviet Union
    "UAR": "EGY",  # United Arab Republic to Egypt
    "FRG": "GER",  # West Germany to Germany
}

@st.cache_data
def get_processed_country_data():
    countries = load_raw_country_data()
    if countries.empty: return pd.DataFrame()
    countries['country'] = countries['country'].replace(country_names_to_change)
    countries['noc'] = countries['noc'].replace(country_NOC_to_change)
    countries = countries.drop_duplicates(subset=['noc'])
    # 1. Save to a temporary clean file
    countries.to_csv('Olympics_Country_Cleaned.csv', index=False)
    # 2. Rename it to overwrite the original
    os.replace('Olympics_Country_Cleaned.csv', 'Olympics_Country.csv')
    return countries


@st.cache_data
def get_processed_main_data():
    df = load_raw_main_data()
    if df.empty: return pd.DataFrame()

    if 'year' in df.columns: df.rename(columns={'year': 'Year'}, inplace=True)
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df.dropna(subset=['Year'])
    df['Year'] = df['Year'].astype(int)

    countries = get_processed_country_data()

    df.loc[df['NOC'] == 'LIB', 'NOC'] = 'LBN'
    df.loc[df['NOC'] == 'ROT', 'NOC'] = 'IOA'
    df['NOC'] = df['NOC'].replace(country_NOC_to_change)

    merged_df = df.merge(countries, left_on='NOC', right_on='noc', how='left')
    merged_df['Team'] = merged_df['country'].fillna(merged_df['Team'])
    df = merged_df.drop(columns=['noc', 'country'])

    return df


def get_name_map():
    # Uses your existing get_processed_country_data to ensure consistency
    df = get_processed_country_data()
    if df.empty: return {}
    return dict(zip(df['noc'], df['country']))


def get_continent_mapping():
    # Loads the new continent data
    df = load_raw_continent_data()
    if df.empty: return {}
    # Maps alpha-3 code to region
    return dict(zip(df['alpha-3'], df['region']))

@st.cache_data
def get_processed_athletics_data():
    df = load_raw_athletics_data()
    if df.empty: return pd.DataFrame()

    df.drop(columns=['Extra'], inplace=True, errors='ignore')
    ref = get_name_map()
    df['Country'] = df['Nationality'].map(ref).fillna(df['Nationality'])

    df['BaseEvent'] = df['Event'].apply(
        lambda e: 'Sprint Hurdles' if 'Hurdles' in str(e) and ('110M' in str(e) or '100M' in str(e)) else str(
            e).replace(' Men', '').replace(' Women', '').strip())

    def parse_result(row):
        val = str(row['Result']).strip().lower()

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
                if any(x in str(row['BaseEvent']) for x in ['Marathon', 'Walk']):
                    return parts[0] * 3600 + parts[1] * 60
                return parts[0] * 60 + parts[1]
            elif len(parts) == 1:
                return parts[0]
        except:
            return np.nan
        return np.nan

    df['NumericResult'] = df.apply(parse_result, axis=1)
    return df


@st.cache_data
def get_processed_medals_data():

    df_medals = load_raw_tally_data()
    df_paris_medals = load_raw_paris_data()
    countries = get_processed_country_data()

    if df_medals.empty: return pd.DataFrame()
    df_medals = df_medals[df_medals['edition'].str.contains('Summer', na=False)].copy()
    df_medals['country_noc'] = df_medals['country_noc'].replace(country_NOC_to_change)
    merged_df = df_medals.merge(countries, left_on='country_noc', right_on='noc', how='left')
    merged_df['country_x'] = merged_df['country_y'].fillna(merged_df['country_x'])
    if 'edition_id' in merged_df.columns:
        df_medals = merged_df.drop(columns=['country_y', 'noc', 'edition_id'])
    else:
        df_medals = merged_df.drop(columns=['country_y', 'noc'])
    df_medals = df_medals.rename(columns={'country_x': 'country'})

    if "2024 Summer Olympics" not in df_medals['edition'].values:
        df_paris_medals['edition'] = '2024 Summer Olympics'
        df_paris_medals['year'] = 2024
        df_paris_medals = df_paris_medals.rename(columns={
            'country_code': 'country_noc',
            'Gold Medal': 'gold',
            'Silver Medal': 'silver',
            'Bronze Medal': 'bronze',
            'Total': 'total'
        })

        df_paris_medals['country_noc'] = df_paris_medals['country_noc'].replace(country_NOC_to_change)
        merged_df = df_paris_medals.merge(countries, left_on='country_noc', right_on='noc', how='left')
        merged_df['country_x'] = merged_df['country_y'].fillna(merged_df['country_x'])
        df_paris_medals = merged_df.rename(columns={'country_x': 'country'})

        columns_order = ['edition', 'year', 'country', 'country_noc', 'gold', 'silver', 'bronze', 'total']
        df_paris_medals = df_paris_medals[columns_order]
        updated_medals_df = pd.concat([df_medals, df_paris_medals], ignore_index=True)
    else:
        updated_medals_df = df_medals
    # 7. Save the new file
    updated_medals_df.to_csv('Olympic_Games_Medal_Tally_Updated.csv', index=False)
    os.replace('Olympic_Games_Medal_Tally_Updated.csv', 'Olympic_Games_Medal_Tally.csv')

    return updated_medals_df


@st.cache_data
def get_processed_gapminder_data():
    # Merges Olympic performance with Gapminder socioeconomic data
    df_main = load_raw_main_data()
    df_medals = get_processed_medals_data()
    if df_main.empty or df_medals.empty: return pd.DataFrame()

    df_medals = df_medals[['year', 'country_noc', 'total']].rename(columns={'total': 'Medals', 'country_noc': 'NOC', 'year': 'Year'})

    if 'year' in df_main.columns: df_main.rename(columns={'year': 'Year'}, inplace=True)
    delegation = df_main.groupby(['Year', 'NOC'])['player_id'].nunique().reset_index().rename(columns={'player_id': 'Delegation_Size'})

    stats = pd.merge(delegation, df_medals, on=['Year', 'NOC'], how='left').fillna({'Medals': 0})
    NOC_TO_CONTINENT = get_continent_mapping()
    stats['Region'] = stats['NOC'].map(NOC_TO_CONTINENT).fillna('Western Europe')

    gapminder = load_gapminder_reference()
    years = range(1896, 2025)
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