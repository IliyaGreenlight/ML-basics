import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

TF_ENABLE_ONEDNN_OPTS=0

file_path = "alta_2025-05-11.xls"
data = pd.read_excel(file_path)

data['Date'] = pd.to_datetime(data['Data'])
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day

close_prices = data[['Date', 'Kurs zamknięcia']].dropna()
close_prices = close_prices.rename(columns={"Kurs zamknięcia": "Close"})

valid_period = close_prices.tail(1000)

train_size = int(len(valid_period) * 0.8)
train_data = valid_period[:train_size]
test_data = valid_period[train_size:]

def create_lagged_data(series, lags):
    lagged_data = pd.DataFrame()
    for lag in range(1, lags + 1):
        lagged_data[f'lag_{lag}'] = series.shift(lag)
    lagged_data['target'] = series
    return lagged_data.dropna()

errors_ar = []
lags_range = range(5, 51, 5)

for lags in lags_range:
    lagged_train = create_lagged_data(train_data['Close'], lags)
    X_train = lagged_train.drop(columns=['target']).values
    y_train = lagged_train['target'].values

    model = AutoReg(y_train, lags=lags).fit()
    predictions = model.predict(start=lags, end=len(y_train)-1)

    error = mean_squared_error(y_train[lags:], predictions)
    errors_ar.append(error)

plt.figure(figsize=(10, 6))
plt.plot(lags_range, errors_ar, marker='o', label='AR Model Errors')
plt.xlabel('Number of Lags')
plt.ylabel('Mean Squared Error')
plt.title('Error Values for Different Numbers of Lags (AR Model)')
plt.legend()
plt.show()

best_lags_ar = lags_range[np.argmin(errors_ar)]
print(f"Best number of lags for AR model: {best_lags_ar}")

errors_rnn = []
scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(valid_period['Close'].values.reshape(-1, 1))

for lags in lags_range:
    lagged_data = create_lagged_data(pd.Series(scaled_data.flatten()), lags)
    X = lagged_data.drop(columns=['target']).values.reshape(-1, lags, 1)
    y = lagged_data['target'].values

    X_train_rnn, y_train_rnn = X[:train_size], y[:train_size]
    X_test_rnn, y_test_rnn = X[train_size:], y[train_size:]

    model_rnn = Sequential([
        LSTM(50, activation='relu', input_shape=(lags, 1)),
        Dense(1)
    ])
    model_rnn.compile(optimizer='adam', loss='mse')

    model_rnn.fit(X_train_rnn, y_train_rnn, epochs=10, verbose=0)
    predictions = model_rnn.predict(X_test_rnn)

    error = mean_squared_error(y_test_rnn, predictions)
    errors_rnn.append(error)

plt.figure(figsize=(10, 6))
plt.plot(lags_range, errors_rnn, marker='o', label='RNN Model Errors', color='orange')
plt.xlabel('Number of Lags')
plt.ylabel('Mean Squared Error')
plt.title('Error Values for Different Numbers of Lags (RNN Model)')
plt.legend()
plt.show()

best_lags_rnn = lags_range[np.argmin(errors_rnn)]
print(f"Best number of lags for RNN model: {best_lags_rnn}")
