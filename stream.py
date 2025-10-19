import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np

st.set_page_config(page_title="Sales Analytics Dashboard", layout="wide")

st.title("📊 Sales Analytics & Forecasting Dashboard")

# Load Data
df = pd.read_csv("outputs/cleaned_sales_data.csv")
st.dataframe(df.head())

# Load Model
model = load_model("models/sales_forecast_model.h5")

# Show historical trend
sales = df.groupby('Date')['Total Sales'].sum()
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(pd.to_datetime(sales.index), sales.values)
ax.set_title('Sales Trend')
st.pyplot(fig)

# Forecast button
if st.button("Predict Next 6 Months"):
    # Dummy forecast logic (replace with actual)
    st.success("✅ Forecast generated")
    st.line_chart(np.random.randint(10000, 20000, 6))
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np

st.set_page_config(page_title="Sales Analytics Dashboard", layout="wide")

st.title("📊 Sales Analytics & Forecasting Dashboard")

# Load Data
df = pd.read_csv("outputs/cleaned_sales_data.csv")
st.dataframe(df.head())

# Load Model
model = load_model("models/sales_forecast_model.h5")

# Show historical trend
sales = df.groupby('Date')['Total Sales'].sum()
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(pd.to_datetime(sales.index), sales.values)
ax.set_title('Sales Trend')
st.pyplot(fig)

# Forecast button
if st.button("Predict Next 6 Months"):
    # Dummy forecast logic (replace with actual)
    st.success("✅ Forecast generated")
    st.line_chart(np.random.randint(10000, 20000, 6))
