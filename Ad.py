
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


df = pd.read_csv('sales_data.csv')
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df.dropna(subset=['Date'], inplace=True)
df['Total Sales'] = df['Units Sold'] * df['Unit Price']

# Group monthly sales
monthly_sales = df.groupby(df['Date'].dt.to_period('M'))['Total Sales'].sum().to_timestamp()


# Convert time series to supervised learning data
def create_dataset(series, window_size=3):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return np.array(X), np.array(y)

# Scale values for stability
sales_values = monthly_sales.values.astype(float)
max_val = np.max(sales_values)
scaled_sales = sales_values / max_val  # Normalization

# Create input-output pairs
window_size = 3
X, y = create_dataset(scaled_sales, window_size)

# Split into train & test
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]


model = Sequential([
    Dense(64, activation='relu', input_shape=(window_size,)),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])


history = model.fit(X_train, y_train, epochs=100, batch_size=4, validation_split=0.2, verbose=0)


loss, mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {mae * max_val:.2f}")

# Make predictions
y_pred = model.predict(X_test)
y_pred_rescaled = y_pred.flatten() * max_val
y_test_rescaled = y_test * max_val

# Forecast next 6 months
future_input = scaled_sales[-window_size:].tolist()
future_forecast = []

for _ in range(6):
    x_input = np.array(future_input[-window_size:]).reshape(1, -1)
    next_pred = model.predict(x_input)[0][0]
    future_forecast.append(next_pred)
    future_input.append(next_pred)

future_forecast = np.array(future_forecast) * max_val


plt.figure(figsize=(12, 6))
plt.plot(monthly_sales.index[-len(y_test):], y_test_rescaled, label='Actual')
plt.plot(monthly_sales.index[-len(y_test):], y_pred_rescaled, label='Predicted', linestyle='--')
plt.title('Actual vs Predicted Monthly Sales')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.legend()
plt.show()

# Plot future forecast
future_dates = pd.date_range(start=monthly_sales.index[-1] + pd.offsets.MonthBegin(1), periods=6, freq='MS')
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales.index, monthly_sales.values, label='Historical')
plt.plot(future_dates, future_forecast, label='Forecast', color='red', marker='o')
plt.title('6-Month Sales Forecast')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.legend()
plt.show()
