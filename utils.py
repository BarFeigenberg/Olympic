# utils.py
# This file contains constant mappings and helper functions for geographical data
import pandas as pd
import streamlit as st

# --- CONSTANT: NOC TO REGION MAPPING ---
# Mapping National Olympic Committees to their respective continents
NOC_TO_CONTINENT = {
    'ALG': 'Africa', 'ANG': 'Africa', 'BDI': 'Africa', 'BEN': 'Africa', 'BOT': 'Africa', 'BUR': 'Africa',
    'CAF': 'Africa', 'CGO': 'Africa', 'CHA': 'Africa', 'CIV': 'Africa', 'CMR': 'Africa', 'COD': 'Africa',
    'COM': 'Africa', 'CPV': 'Africa', 'DJI': 'Africa', 'EGY': 'Africa', 'ERI': 'Africa', 'ETH': 'Africa',
    'GAB': 'Africa', 'GAM': 'Africa', 'GBS': 'Africa', 'GEQ': 'Africa', 'GHA': 'Africa', 'GUI': 'Africa',
    'KEN': 'Africa', 'LBR': 'Africa', 'LBA': 'Africa', 'LES': 'Africa', 'MAD': 'Africa', 'MAR': 'Africa',
    'MAW': 'Africa', 'MLI': 'Africa', 'MOZ': 'Africa', 'MRI': 'Africa', 'MTN': 'Africa', 'NAM': 'Africa',
    'NGR': 'Africa', 'NIG': 'Africa', 'RSA': 'Africa', 'RWA': 'Africa', 'SEN': 'Africa', 'SEY': 'Africa',
    'SLE': 'Africa', 'SOM': 'Africa', 'SSD': 'Africa', 'STP': 'Africa', 'SUD': 'Africa', 'SWZ': 'Africa',
    'TAN': 'Africa', 'TOG': 'Africa', 'TUN': 'Africa', 'UGA': 'Africa', 'ZAM': 'Africa', 'ZIM': 'Africa',
    'AFG': 'Asia', 'BRN': 'Asia', 'BAN': 'Asia', 'BHU': 'Asia', 'BRU': 'Asia', 'CAM': 'Asia', 'CHN': 'Asia',
    'HKG': 'Asia', 'INA': 'Asia', 'IND': 'Asia', 'IRI': 'Asia', 'IRQ': 'Asia', 'JPN': 'Asia', 'JOR': 'Asia',
    'KAZ': 'Asia', 'KGZ': 'Asia', 'KOR': 'Asia', 'KSA': 'Asia', 'KUW': 'Asia', 'LAO': 'Asia', 'LIB': 'Asia',
    'MAC': 'Asia', 'MAS': 'Asia', 'MDV': 'Asia', 'MGL': 'Asia', 'MYA': 'Asia', 'NEP': 'Asia', 'OMA': 'Asia',
    'PAK': 'Asia', 'PHI': 'Asia', 'PLE': 'Asia', 'PRK': 'Asia', 'QAT': 'Asia', 'SIN': 'Asia', 'SRI': 'Asia',
    'SYR': 'Asia', 'THA': 'Asia', 'TJK': 'Asia', 'TKM': 'Asia', 'TPE': 'Asia', 'UAE': 'Asia', 'UZB': 'Asia',
    'VIE': 'Asia', 'YEM': 'Asia',
    'ALB': 'Western Europe', 'AND': 'Western Europe', 'ARM': 'Western Europe', 'AUT': 'Western Europe', 'AZE': 'Western Europe', 'BEL': 'Western Europe',
    'BIH': 'Western Europe', 'BLR': 'Western Europe', 'BUL': 'Western Europe', 'CRO': 'Western Europe', 'CYP': 'Western Europe', 'CZE': 'Western Europe',
    'DEN': 'Western Europe', 'ESP': 'Western Europe', 'EST': 'Western Europe', 'FIN': 'Western Europe', 'FRA': 'Western Europe', 'GBR': 'Western Europe',
    'GEO': 'Western Europe', 'GER': 'Western Europe', 'GRE': 'Western Europe', 'HUN': 'Western Europe', 'IRL': 'Western Europe', 'ISL': 'Western Europe',
    'ISR': 'Western Europe', 'ITA': 'Western Europe', 'KOS': 'Western Europe', 'LAT': 'Western Europe', 'LIE': 'Western Europe', 'LTU': 'Western Europe',
    'LUX': 'Western Europe', 'MDA': 'Western Europe', 'MKD': 'Western Europe', 'MLT': 'Western Europe', 'MNE': 'Western Europe', 'MON': 'Western Europe',
    'NED': 'Western Europe', 'NOR': 'Western Europe', 'POL': 'Western Europe', 'POR': 'Western Europe', 'ROU': 'Western Europe', 'RUS': 'Western Europe',
    'SLO': 'Western Europe', 'SMR': 'Western Europe', 'SRB': 'Western Europe', 'SUI': 'Western Europe', 'SVK': 'Western Europe', 'SWE': 'Western Europe',
    'TUR': 'Western Europe', 'UKR': 'Western Europe', 'URS': 'Western Europe', 'GDR': 'Western Europe', 'FRG': 'Western Europe', 'TCH': 'Western Europe', 'YUG': 'Western Europe', 'EUN': 'Western Europe',
    'ANT': 'Americas', 'ARG': 'Americas', 'ARU': 'Americas', 'BAH': 'Americas', 'BAR': 'Americas', 'BER': 'Americas',
    'BIZ': 'Americas', 'BOL': 'Americas', 'BRA': 'Americas', 'CAN': 'Americas', 'CAY': 'Americas', 'CHI': 'Americas',
    'COL': 'Americas', 'CRC': 'Americas', 'CUB': 'Americas', 'DMA': 'Americas', 'DOM': 'Americas', 'ECU': 'Americas',
    'ESA': 'Americas', 'GRN': 'Americas', 'GUA': 'Americas', 'GUY': 'Americas', 'HAI': 'Americas', 'HON': 'Americas',
    'ISV': 'Americas', 'IVB': 'Americas', 'JAM': 'Americas', 'LCA': 'Americas', 'MEX': 'Americas', 'NCA': 'Americas',
    'PAN': 'Americas', 'PAR': 'Americas', 'PER': 'Americas', 'PUR': 'Americas', 'SKN': 'Americas', 'SUR': 'Americas',
    'TTO': 'Americas', 'URU': 'Americas', 'USA': 'Americas', 'VEN': 'Americas', 'VIN': 'Americas',
    'ASA': 'Oceania', 'AUS': 'Oceania', 'COK': 'Oceania', 'FIJ': 'Oceania', 'FSM': 'Oceania', 'GUM': 'Oceania',
    'KIR': 'Oceania', 'MHL': 'Oceania', 'NRU': 'Oceania', 'NZL': 'Oceania', 'PLW': 'Oceania', 'PNG': 'Oceania',
    'SAM': 'Oceania', 'SOL': 'Oceania', 'TGA': 'Oceania', 'TUV': 'Oceania', 'VAN': 'Oceania'
}

@st.cache_data
def get_name_map():
    # Loads NOC code to full country name mapping from CSV
    try:
        ref = pd.read_csv("Olympics_Country.csv")
        return ref.set_index('noc')['country'].to_dict()
    except Exception as e:
        print(f"Warning: Could not load name map - {e}")
        return {}