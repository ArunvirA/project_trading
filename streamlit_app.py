import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts.models import Theta
from darts import TimeSeries
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.title("Data Led Trading: Predict, Profit, Prevail")
