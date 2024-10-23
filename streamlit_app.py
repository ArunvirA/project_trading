import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts.models import Theta
from darts import TimeSeries
import plotly.express as px
import plotly.graph_objects as go

st.title('EUR/USD Close Price Forecast using Theta Model')
st.markdown("""This app forecasts EUR/USD close prices using machine learning specifically the Theta model.""")

df = pd.read_csv('eurusd_24h.csv')
st.dataframe(df.head())

df = df.sort_values(by='Date_date', ascending=True)
# df.set_index('Date_date', inplace=True, drop=False)
df = df[['Date_date', 'open', 'high', 'low', 'close', 'vol']]
df['Date_date'] = pd.to_datetime(df['Date_date'])

df['close'] = df['close'].interpolate()

# train, test = y[:int(len(y)*0.8)], y[int(len(y)*0.8):]

# y2 = y
full_date_range = pd.date_range(start=df['Date_date'].min(), end=df['Date_date'].max(), freq='D')
# full_date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
df.set_index('Date_date', inplace=True, drop=False)
df_full = df[['open', 'high', 'low', 'close', 'vol']].reindex(full_date_range)
df_full['close'] = df_full['close'].fillna(method='ffill')
y_full = df_full['close'] #.fillna(method='ffill')

df_full.reset_index(inplace=True)
df_full.rename(columns={'index': 'Date_date'}, inplace=True)

y_ts = TimeSeries.from_series(y_full, fill_missing_dates=True, freq="D")
train_ts, test_ts = y_ts[:int(len(y_ts)*0.8)], y_ts[int(len(y_ts)*0.8):]

# st.write(df[['open', 'high', 'low', 'close', 'vol']])
# st.write(df_full)
# st.write(y_ts)
# st.write(y_full)

theta = Theta()
theta.fit(train_ts)
forecast_values = theta.predict(len(test_ts))

# st.subheader("Forecast Values")
# st.write(forecast_values.pd_dataframe())

# st.write(forecast_values)

st.subheader("Plotting Forecast Results")
fig, ax = plt.subplots(figsize=(12, 6))
train_ts.plot(ax=ax, label='Train', lw=2)
test_ts.plot(ax=ax, label='Test', lw=2)
forecast_values.plot(ax=ax, label='Forecast', lw=2)
plt.xlabel('Date')
plt.title('EUR/USD Close Price Forecast using Theta')
plt.legend()

st.pyplot(fig)

mae_theta = 0.10251161515453634
mse_theta = 0.008376373176977219
rmse_theta = 0.0915225282483893


st.subheader("Model Evaluation:")

col1, col2, col3 = st.columns(3)
col1.metric("MAE", "0.10251")
col2.metric("MSE", "0.00837")
col3.metric("RMSE", "0.09152")


# def get_plotly_data():

#     z_data = pd.read_csv('eurusd_24h.csv')
#     z = z_data.values
#     sh_0, sh_1 = z.shape
#     x, y = np.linspace(0, 1, sh_0), np.linspace(0, 1, sh_1)
#     return x, y, z
# st.write(train_ts)
# st.write(train_ts.values())
# st.write(len(train_ts.values()))

# x, y, z = get_plotly_data()
# train_line = px.line(x=df_full.index[:len(train_ts.values())],y=train_ts.values())
# test_line = px.line(x=df_full.index[-len(test_ts.values())+1:],y=test_ts.values())
# forecast_line = px.line(x=df_full.index[-len(test_ts.values())+1:],y=forecast_values.values())

# fig = go.Figure(data=[train_line,test_line,forecast_line])
# fig.update_layout(title='EUR/USD predictions', autosize=False, width=800, height=800, margin=dict(l=40, r=40, b=40, t=40))
# st.plotly_chart(fig)

if st.button('More 🎈🎈🎈 please!'):
    st.balloons()

df_full['Date_date'] = df_full['Date_date'].dt.tz_localize('UTC', ambiguous='infer')  # or remove timezone
fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(x=train_ts.index, y=train_ts, mode='lines', name='Train', line=dict(width=2)))
fig.add_trace(go.Scatter(x=test_ts.index, y=test_ts, mode='lines', name='Test', line=dict(width=2)))
fig.add_trace(go.Scatter(x=forecast_values.index, y=forecast_values, mode='lines', name='Forecast', line=dict(width=2, dash='dash')))

# Update layout
fig.update_layout(title="EUR/USD Close Price Forecast using Theta",
                  xaxis_title="Date",
                  yaxis_title="Close Price",
                  legend=dict(x=0, y=1))

# Show the figure in Streamlit
st.plotly_chart(fig)







