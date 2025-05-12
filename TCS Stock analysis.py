import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tqdm import tqdm

df = pd.read_csv('TCS_stock_history.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)
print(df.info())

df.fillna(method='ffill', inplace=True)
df.describe()

plt.figure(figsize=(14, 6))
plt.plot(df['Date'], df['Close'], label='Close Price', color='b')
plt.title('TCS Stock Close Price Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

df['MA50'] = df['Close'].rolling(window=50).mean()
df['MA200'] = df['Close'].rolling(window=200).mean()

plt.figure(figsize=(14, 6))
plt.plot(df['Date'], df['Close'], label='Close Price')
plt.plot(df['Date'], df['MA50'], label='50-Day MA')
plt.plot(df['Date'], df['MA200'], label='200-Day MA')
plt.title('TCS Stock with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Day_of_Week'] = df['Date'].dt.dayofweek
df['Prev_Close'] = df['Close'].shift(1)
df.dropna(inplace=True)

features = ['Open', 'High', 'Low', 'Volume', 'Prev_Close', 'Day_of_Week', 'Month']
X = df[features]
y = df['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_test)

print("Linear Regression MSE:", mean_squared_error(y_test, y_pred))
print("Linear Regression R2 Score:", r2_score(y_test, y_pred))

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Close']])

X_lstm, y_lstm = [], []
window_size = 60
for i in range(window_size, len(scaled_data)):
    X_lstm.append(scaled_data[i-window_size:i, 0])
    y_lstm.append(scaled_data[i, 0])

X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))

lstm_model = Sequential()
lstm_model.add(LSTM(50, return_sequences=False, input_shape=(X_lstm.shape[1], 1)))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

lstm_model.fit(X_lstm, y_lstm, epochs=10, batch_size=32)

test_data = scaled_data[-(window_size + int(len(df)*0.2)):]
X_test_lstm = []
for i in range(window_size, len(test_data)):
    X_test_lstm.append(test_data[i-window_size:i, 0])

X_test_lstm = np.array(X_test_lstm)
X_test_lstm = np.reshape(X_test_lstm, (X_test_lstm.shape[0], X_test_lstm.shape[1], 1))

lstm_predictions = lstm_model.predict(X_test_lstm)
lstm_predictions = scaler.inverse_transform(lstm_predictions)

plt.figure(figsize=(14, 6))
plt.plot(df['Date'][-len(lstm_predictions):], df['Close'][-len(lstm_predictions):], label='Actual Close Price')
plt.plot(df['Date'][-len(lstm_predictions):], lstm_predictions, label='Predicted Close Price (LSTM)', color='red')
plt.title('Actual vs Predicted Close Price (LSTM)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

lstm_mae = mean_absolute_error(df['Close'][-len(lstm_predictions):], lstm_predictions)
print("LSTM Mean Absolute Error:", lstm_mae)
