import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts.models import Theta
from darts import TimeSeries
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.title('EUR/USD Close Price Forecast using Theta Model')
st.markdown("""This app forecasts EUR/USD close prices using machine learning specifically the Theta model.""")

df = pd.read_csv('eurusd_24h.csv')
st.dataframe(df.head())

df = df.sort_values(by='Date_date', ascending=True)
df.reset_index(inplace=True, drop=True)
df = df[['Date_date', 'open', 'high', 'low', 'close', 'vol']]

y = df['close'].interpolate()

train, test = y[:int(len(y)*0.8)], y[int(len(y)*0.8):]

full_date_range = pd.date_range(start=df['Date_date'].min(), end=df['Date_date'].max(), freq='D')
y_full = pd.DataFrame(y.reindex(full_date_range), columns=['close'])
y_full['close'] = y_full['close'].fillna(method='ffill')

y_ts = TimeSeries.from_series(y_full['close'], fill_missing_dates=True, freq="D")
train_ts, test_ts = y_ts[:int(len(y_ts)*0.8)], y_ts[int(len(y_ts)*0.8):]

st.write("Training Theta model...")
theta = Theta()
theta.fit(train_ts)
forecast_values = theta.predict(len(test_ts))

st.write("Plotting forecast results...")
fig, ax = plt.subplots(figsize=(12, 6))
train_ts.plot(label='Train', lw=2, ax=ax)
test_ts.plot(label='Test', lw=2, ax=ax)
forecast_values.plot(label='Forecast', lw=2, ax=ax)
plt.xlabel('Date')
plt.title('EUR/USD Close Price Forecast using Theta')

st.pyplot(fig)











