# IDX Stock Price Predictor

A machine learning-based stock price prediction system for the Indonesia Stock Exchange (IDX), using streamlit interface.

## Features

- Historical data analysis from IDX
- Linear Regression model with 5-day prediction window
- Interactive web interfce for stock price predictions
- Support for all IDX-listed tickers

## Prerequisites

- Python 3.8+
- pandas
- numpy
- scikit-learn
- streamlit

## Installation

```bash
git clone https://github.com/ni-zav/stock-predict-idx.git
cd stock-predict-idx
pip install -r requirements.txt
```

## Dataset

Place `idx.csv` in the project root directroy. The dataset should contain:
- Date
- Ticker
- Open, High, Low, Close
- Adj Close
- Volume

## Usage

1. Train the model:
```bash
jupyter notebook notebook.ipynb
```

2. Launch the web application:
```bash
streamlit run app.py
```

## Model Evaluation

The application uses Mean Squared Error (MSE) to evaluate prediction accuracy:
- MSE is displayed for each ticker
- Lower MSE values indicate better model performance
- Model performance varies by stock volatility and trading volume

## Project Structure

```
stock-predict-idx/
├── notebook.ipynb     # Data processing and model training
├── app.py            # Streamlit web application
├── models.pkl        # Trained models (generated)
└── idx.csv          # Historical data (not included)
```

## Model Detials

- Uses Linear Regresssion algorithm
- Features: Previous 5 days' closing prices
- Training/Test spilt: 80/20
- No shuffle to maintain time series integrity
- Predictions are made for next business day

## Future Enhancements

- Advanced models (LSTM, Random Forest)
- Technical indicators integration
- Real-time data updates
- Enhanced visualization
- Improved error handling

## Limitations

- Limited to linear relationships
- Does not account for market sentiment
- Predictions based soleyl on historical prices
- Past performance doesn't guarantee future results

## Disclaimer

This project is for educational purposes only. The predictions should not be used for actual trading decisions. Users must comply with all applicable financial regulations and data usage policies.

