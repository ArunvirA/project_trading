import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pmdarima import auto_arima
from darts.models import Theta
from darts import TimeSeries
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.title("Data Led Trading: Predict, Profit, Prevail")
