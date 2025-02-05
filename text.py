import pandas as pd
import numpy as np
import os
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def load_data():
    df = pd.read_csv('idx.csv', parse_dates=['Date'])
    df.sort_values(['Ticker', 'Date'], inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

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

def get_last_5_closing_prices(ticker, df):
    ticker_data = df[df['Ticker'] == ticker].copy()
    ticker_data.sort_values('Date', inplace=True)
    return ticker_data['Close'].iloc[-5:].values

def predict_next_price(ticker, df, models):
    if ticker not in models:
        return None, "No trained model available for this ticker."
        
    mse = models[ticker]['mse']
    last_5_prices = get_last_5_closing_prices(ticker, df)
    
    if len(last_5_prices) != 5:
        return None, "Not enough data to make a prediction."
        
    model = models[ticker]['model']
    # Create DataFrame with proper feature names
    input_features = pd.DataFrame(
        [last_5_prices], 
        columns=[f'lag_{i}' for i in range(1, 6)]
    )
    predicted_price = model.predict(input_features)[0]
    
    return predicted_price, mse

# Usage example:
if __name__ == "__main__":
    df = load_data()
    test_ticker = 'GOTO'
    
    # Load or train models
    models_file = 'models.pkl'
    if os.path.exists(models_file):
        with open(models_file, 'rb') as f:
            models = pickle.load(f)
    else:
        models = {}
        for ticker in df['Ticker'].unique():
            model, mse = train_model_for_ticker(ticker, df)
            if model:
                models[ticker] = {'model': model, 'mse': mse}
        with open(models_file, 'wb') as f:
            pickle.dump(models, f)
            
    # Make prediction
    predicted_price, mse = predict_next_price(test_ticker, df, models)
    if predicted_price is not None:
        print(f"MSE: {mse:.4f}")
        print(f"Last 5 prices: {get_last_5_closing_prices(test_ticker, df)}")
        print(f"Predicted next price: {predicted_price:.2f}")
