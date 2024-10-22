import streamlit as st
import pandas as pd
import numpy as np
from darts.models import Theta
from darts import TimeSeries
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.title('EUR/USD Close Price Forecast using Theta Model')
st.write("""This app forecasts EUR/USD close prices using the Theta model. You can upload your own dataset as a CSV file for analysis.""")

df = pd.read_csv('eurusd_24h.csv')
st.dataframe(df.head())
