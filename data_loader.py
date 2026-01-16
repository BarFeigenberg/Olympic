# data_loader.py
# This module handles raw data loading from CSV/TSV files with Streamlit caching
# Ensures performance by caching file reads and gracefully handling missing files

import pandas as pd
import streamlit as st
import plotly.express as px


# --- Games / Olympics Files ---
@st.cache_data
def load_raw_games_data():
    """Load raw Olympics Games metadata from CSV."""
    try:
        return pd.read_csv('Olympics_Games.csv')
    except FileNotFoundError:
        return pd.DataFrame()


@st.cache_data
def load_raw_tally_data():
    """Load medal tally CSV (all-time or per edition)."""
    try:
        return pd.read_csv('Olympic_Games_Medal_Tally.csv')
    except FileNotFoundError:
        return pd.DataFrame()


@st.cache_data
def load_raw_main_data():
    """Load the main Olympics dataset (events/athletes/medals)."""
    try:
        return pd.read_csv("olympics_dataset.csv")
    except FileNotFoundError:
        return pd.DataFrame()


@st.cache_data
def load_raw_country_data():
    """Load mapping of NOCs to country names."""
    try:
        return pd.read_csv("Olympics_Country.csv")
    except FileNotFoundError:
        return pd.DataFrame()


# --- Athletics Specific Data ---
@st.cache_data
def load_raw_athletics_data():
    """
    Load athletics results with fixed column names:
    Gender, Event, Location, Year, Medal, Name, Nationality, Result, Extra
    Skips the CSV header row (assumes first row is header in file)
    """
    try:
        return pd.read_csv(
            'results.csv',
            names=['Gender', 'Event', 'Location', 'Year', 'Medal', 'Name', 'Nationality', 'Result', 'Extra'],
            skiprows=1
        )
    except FileNotFoundError:
        return pd.DataFrame()


@st.cache_data
def load_raw_paris_data():
    """Load precomputed medal totals for Paris 2024."""
    try:
        return pd.read_csv('medals_total_paris.csv')
    except FileNotFoundError:
        return pd.DataFrame()


# --- Gapminder Reference Data ---
@st.cache_data
def load_gapminder_reference():
    """Load built-in Plotly Gapminder dataset (life expectancy, population, GDP)."""
    return px.data.gapminder()


# --- Continent Mapping ---
@st.cache_data
def load_raw_continent_data():
    """Load continent mapping for countries/NOCs."""
    try:
        return pd.read_csv("continent_data.csv")
    except FileNotFoundError:
        return pd.DataFrame()


# --- Population Data ---
@st.cache_data
def load_historical_population_data():
    """Load historical population data (e.g., Our World in Data)."""
    try:
        return pd.read_csv('population.csv')
    except FileNotFoundError:
        return pd.DataFrame()


@st.cache_data
def load_2024_population_data():
    """
    Load 2024 population update.
    Note: File is TSV (.xls extension but tab-separated), handles thousands separator and encoding.
    """
    try:
        return pd.read_csv('population2024.xls', sep='\t', thousands=',', encoding='latin-1')
    except (FileNotFoundError, pd.errors.ParserError):
        return pd.DataFrame()


# --- Life Expectancy Data ---
@st.cache_data
def load_life_expectancy_data():
    """Load life expectancy CSV with Country and Year data."""
    try:
        return pd.read_csv('lex.csv')
    except FileNotFoundError:
        return pd.DataFrame()


# --- Athlete Bio Data ---
@st.cache_data
def load_athlete_bio_data():
    """Load athlete biography information (height, weight, birthdate, etc.)."""
    try:
        return pd.read_csv("Olympic_Athlete_Bio.csv")
    except FileNotFoundError:
        return pd.DataFrame()
