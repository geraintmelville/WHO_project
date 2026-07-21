"""
Life expectancy dashboard by country
================================
Streamlit dashboard for WHO — allows users to calculate an estimate of life expectancy in their country, based on a number of features.

Run with:  streamlit run app.py
"""


import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np