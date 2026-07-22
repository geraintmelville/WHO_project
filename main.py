import streamlit as st
import pandas as pd
import numpy as np
from app import *
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv('Life Expectancy Data.csv')

st.title('WHO Report')

st.header('minimal data model')

st.write(least_information(data))
st.write(li_pipeline(data))
