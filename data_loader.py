# data_loader.py
# This module handles raw data loading from CSV files with caching for performance
import pandas as pd
import streamlit as st
import plotly.express as px

@st.cache_data
def load_raw_games_data():
    try:
        return pd.read_csv('Olympics_Games.csv')
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_data
def load_raw_tally_data():
    try:
        return pd.read_csv('Olympic_Games_Medal_Tally.csv')
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_data
def load_raw_main_data():
    try:
        return pd.read_csv("olympics_dataset.csv")
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_data
def load_raw_country_data():
    try:
        return pd.read_csv("Olympics_Country.csv")
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_data
def load_raw_athletics_data():
    try:
        # Loading athletics results with specific column names
        return pd.read_csv('results.csv', names=['Gender', 'Event', 'Location', 'Year', 'Medal', 'Name', 'Nationality', 'Result', 'Extra'], skiprows=1)
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_data
def load_raw_paris_data():
    try:
        return pd.read_csv('medals_total_paris.csv')
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_data
def load_gapminder_reference():
    # Built-in plotly data for life expectancy and population comparisons
    return px.data.gapminder()

@st.cache_data
def load_raw_continent_data():
    try:
        return pd.read_csv("continent_data.csv")
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_data
def load_athlete_bio_data():
    try:
        return pd.read_csv("Olympic_Athlete_Bio.csv")
    except FileNotFoundError:
        return pd.DataFrame()
