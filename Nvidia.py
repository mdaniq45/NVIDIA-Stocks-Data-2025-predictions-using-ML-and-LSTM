
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

# Set the title of the app
st.title("NVIDIA Stock Price Prediction with LSTM")

# File uploader widget for CSV file
uploaded_file = st.file_uploader(r"C:\Users\hi\Downloads\NVDA.csv", type=["csv"])

if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)
    
    # Convert 'Date' to datetime and sort
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Display a line chart of the closing prices if the user wants
    if st.checkbox("Show Closing Price Chart"):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['Date'], df['Close'], color='blue', label="Close")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.set_title("NVIDIA Stock Closing Price")
        ax.legend()
        st.pyplot(fig)

    # Preprocessing: We use only the 'Close' column for prediction.
    data = df[['Close']].values

    # Normalize the data for LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create sequences for the LSTM model
    st.subheader("LSTM Sequence Configuration")
    time_step = st.slider("Select sequence length (number of days)", min_value=30, max_value=90, value=60)

    def create_sequences(dataset, time_step):
        X, y = [], []
        for i in range(len(dataset) - time_step):
            X.append(dataset[i:(i+time_step), 0])
            y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Split into training and testing sets (80/20 split)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    st.write(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # Build the LSTM model
    st.subheader("LSTM Model Training")
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Allow the user to set the number of epochs
    epochs = st.number_input("Select number of epochs", min_value=1, max_value=100, value=20)
    
    st.write("Training the model, please wait...")
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, 
                        validation_data=(X_test, y_test), verbose=0)
    st.success("Model training completed!")

    # Make predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Plot Actual vs Predicted prices
    st.subheader("Predictions vs Actual Closing Prices")
    # Note: The test set starts at index (train_size + time_step)
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    dates_for_plot = df['Date'][train_size + time_step:]
    ax2.plot(dates_for_plot, y_test_actual, label="Actual Price", color='blue')
    ax2.plot(dates_for_plot, predictions, label="Predicted Price", color='red')
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Price (USD)")
    ax2.set_title("LSTM Stock Price Prediction")
    ax2.legend()
    st.pyplot(fig2)

    # Evaluate model performance
    mse = mean_squared_error(y_test_actual, predictions)
    mae = mean_absolute_error(y_test_actual, predictions)
    rmse = math.sqrt(mse)

    st.subheader("Model Performance Metrics")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
