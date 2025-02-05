# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('idx.csv', parse_dates=['Date'])
    df.sort_values(['Ticker', 'Date'], inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

df = load_data()

# Function to train the model for a ticker
def train_model_for_ticker(ticker, df):
    ticker_data = df[df['Ticker'] == ticker].copy()
    ticker_data.sort_values('Date', inplace=True)
    ticker_data.reset_index(drop=True, inplace=True)
    
    if len(ticker_data) < 10:
        return None, None

    for lag in range(1, 6):
        ticker_data[f'lag_{lag}'] = ticker_data['Close'].shift(lag)
    ticker_data.dropna(inplace=True)
    
    X = ticker_data[[f'lag_{lag}' for lag in range(1, 6)]]
    y = ticker_data['Close']
    
    if len(X) < 10:
        return None, None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    return model, mse

# Function to get last 5 closing prices and dates
def get_last_5_closing_prices_and_dates(ticker, df):
    ticker_data = df[df['Ticker'] == ticker].copy()
    ticker_data.sort_values('Date', inplace=True)
    closing_prices = ticker_data['Close'].iloc[-5:].values
    dates = ticker_data['Date'].iloc[-5:].values
    return closing_prices, dates

# Build or load models
models_file = 'models.pkl'

if os.path.exists(models_file):
    with open(models_file, 'rb') as f:
        models = pickle.load(f)
else:
    tickers = df['Ticker'].unique()
    models = {}
    for ticker in tickers:
        model, mse = train_model_for_ticker(ticker, df)
        if model:
            models[ticker] = {'model': model, 'mse': mse}
    with open(models_file, 'wb') as f:
        pickle.dump(models, f)

# Streamlit App
st.title('IDX Stock Price Predictor')

# Check if any models are available
if not models:
    st.write('No models are available. Please ensure that models are trained and available.')
else:
    # Ticker selection
    ticker = st.selectbox('Select a Ticker', sorted(models.keys()))
    
    # Display MSE
    mse = models[ticker]['mse']
    st.write(f'Model Mean Squared Error: {mse:.4f}')
    st.write('''
    *Mean Squared Error (MSE) indicates the model's prediction accuracy:*
    - Lower MSE values indicate better predictions
    - MSE is calculated as the average of squared differences between predicted and actual values
    - The closer to 0, the more accurate the model
    ''')
    
    # Get last 5 closing prices and dates
    last_5_prices, last_5_dates = get_last_5_closing_prices_and_dates(ticker, df)
    
    # Ensure there are enough data points
    if len(last_5_prices) == 5:
        st.write('Last 5 Closing Prices:')
        for date, price in zip(last_5_dates, last_5_prices):
            formatted_date = pd.Timestamp(date).strftime('%Y-%m-%d')
            st.write(f'Date: {formatted_date}, Closing Price: {price}')
        
        # Predict the next closing price
        model = models[ticker]['model']
        input_features = last_5_prices.reshape(1, -1)
        predicted_price = model.predict(input_features)[0]
        
        # Predict the date for the next closing price
        last_date = pd.Timestamp(last_5_dates[-1])
        next_date = last_date + pd.offsets.BDay()
        
        st.write(f'Predicted Next Closing Price for **{ticker}** on {next_date.strftime("%Y-%m-%d")}: **{predicted_price:.2f}**')
        
        # Ensure data sensitivity is considered
        # Be cautious when displaying financial predictions; data may be sensitive.
    else:
        st.write('Not enough data to make a prediction for this ticker.')