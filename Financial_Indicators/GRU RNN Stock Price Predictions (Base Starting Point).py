import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import Huber
from tensorflow.keras.layers import Dropout
import sys

# System encoding and Presets
sys.stdout.reconfigure(encoding='utf-8')

# Variable Inputs
stock = "BJDX"
sequence_length = 60



# Step 1. Download data from Yahoo Finance and apply technical indicators
def download_ticker(stock):
    ticker = yf.Ticker(stock).history(period="10y", interval="1d")
    if ticker.empty:
        ticker = yf.Ticker(stock).history(period='max', interval="1d")
    ticker = ticker[['Close']]

    def moving_average(data, window_size):
        return data.rolling(window=window_size).mean()

    # Add the moving average indicators to the data
    ticker['MA10'] = moving_average(ticker['Close'], 10)
    ticker['MA20'] = moving_average(ticker['Close'], 20)
    ticker['MA50'] = moving_average(ticker['Close'], 50)

    # Calculate the exponential moving average
    def exponential_moving_average(data, window_size):
        return data.ewm(span=window_size, adjust=False).mean()
    
    # Add the MACD technical indicator which is the difference between the 12 and 26 day EMA
    ticker['MACD'] = exponential_moving_average(ticker['Close'], 12) - exponential_moving_average(ticker['Close'], 26)




    ticker = ticker.dropna()
    return ticker


print(download_ticker(stock))

# Data preprocessing 
def preprocess_ticker(ticker, sequence_length, split_ratio=0.8):
    # Normalize the data
    scaler = MinMaxScaler()
    ticker_scaled = scaler.fit_transform(ticker)

    # Create sequences
    sequences, targets = [], []
    for i in range(len(ticker_scaled) - sequence_length):
        sequences.append(ticker_scaled[i:i + sequence_length])
        targets.append(ticker_scaled[i + sequence_length])
    
    sequences, targets = np.array(sequences), np.array(targets)


    # Split into training and test sets
    split_index = int(len(sequences) * split_ratio)
    X_train, X_test = sequences[:split_index], sequences[split_index:]
    Y_train, Y_test = targets[:split_index], targets[split_index:]

    return X_train, Y_train, X_test, Y_test, scaler


sequence_length = 60 # This means we are grabbing 60 timeseries points (days) to predict the next day stock price
X_train, Y_train, X_test, Y_test, scaler = preprocess_ticker(download_ticker(stock), sequence_length)


# Build the RNN model
def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Bidirectional(GRU(50, return_sequences=True)),
        
        Bidirectional(GRU(50)),
        
        Dense(1) # output layer for regression
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss=Huber())
    return model


input_shape = (X_train.shape[1], X_train.shape[2])
model = build_model(input_shape)


# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_test, Y_test), verbose=1, callbacks=[early_stopping])


# Evaluation and Prediction
# Plot and training loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()


# Predict on test set
predicted_stock_price = model.predict(X_test)
predicted_stock_price = scaler.inverse_transform(np.concatenate([predicted_stock_price, np.zeros((predicted_stock_price.shape[0], 2))], axis=1))[:, 0]
actual_stock_price = scaler.inverse_tranform(np.concatenate(np.concatenate([Y_test.reshape(-1, 1), np.zeroes((Y_test.shape[0], 2))], axis=1))[:, 0])


# Plot the actual vs predicted stock price
plt.figure(figsize=(12,6))
plt.plot(actual_stock_price, color='gray', label='Actual Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Stock Price')
plt.title(f"{stock} Stock Price Prediction")
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.legend()
plt.show()