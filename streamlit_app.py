import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts.models import Theta
from darts import TimeSeries

st.title('EUR/USD Close Price Forecast using Theta Model')
st.markdown("""This app forecasts EUR/USD close prices using machine learning specifically the Theta model.""")

df = pd.read_csv('eurusd_24h.csv')
st.dataframe(df.head())

df = df.sort_values(by='Date_date', ascending=True)
df.reset_index(inplace=True, drop=True)
df = df[['Date_date', 'open', 'high', 'low', 'close', 'vol']]

y = df['close'].interpolate()

train, test = y[:int(len(y)*0.8)], y[int(len(y)*0.8):]

y2 = y
full_date_range = pd.date_range(start=df['Date_date'].min(), end=df['Date_date'].max(), freq='D')
y_full = pd.DataFrame(y.reindex(full_date_range), columns=['close'])
y_full['close'] = y_full['close'].fillna(method='ffill')

y_ts = TimeSeries.from_series(y_full['close'], fill_missing_dates=True, freq="D")
train_ts, test_ts = y_ts[:int(len(y_ts)*0.8)], y_ts[int(len(y_ts)*0.8):]

theta = Theta()
theta.fit(train_ts)
forecast_values = theta.predict(len(test_ts))

st.subheader("Plotting forecast results")
fig, ax = plt.subplots(figsize=(12, 6))
train_ts.plot(label='Train', lw=2, ax=ax)
test_ts.plot(label='Test', lw=2, ax=ax)
forecast_values.plot(label='Forecast', lw=2, ax=ax)
plt.xlabel('Date')
plt.title('EUR/USD Close Price Forecast using Theta')

st.pyplot(fig)

mae_theta = 0.0798717967958772
mse_theta = 0.008376373176977219
rmse_theta = 0.0915225282483893

st.subheader("Model Evaluation:")
st.markdown(f"<p style='color: white; font-size: 18px;'>MAE: {mae_theta:.5f}</p>", unsafe_allow_html=True)
st.markdown(f"<p style='color: white; font-size: 18px;'>MSE: {mse_theta:.5f}</p>", unsafe_allow_html=True)
st.markdown(f"<p style='color: white; font-size: 18px;'>RMSE: {rmse_theta:.5f}</p>", unsafe_allow_html=True)










