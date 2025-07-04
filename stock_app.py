import streamlit as st
import pandas as pd
import yfinance as yf
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

st.title('Stock Price Analytics & Predictions')
st.sidebar.info('Welcome to the Stock Price Prediction App')
# Changed this line as per our discussion to be more general or removed if using README for credit
st.sidebar.info("Developed by: Mohammed Abdul Zeeshan") # Or comment out if relying solely on README

def main():
    """Main function to control the app flow based on sidebar selection."""
    option = st.sidebar.selectbox('Make a choice', ['Visualize', 'Recent Data', 'Predict'])
    if option == 'Visualize':
        tech_indicators()
    elif option == 'Recent Data':
        dataframe()
    else:
        predict()

@st.cache_resource
def download_data(op, start_date, end_date):
    """
    Downloads stock data using yfinance and caches it for performance.
    Args:
        op (str): Stock ticker symbol.
        start_date (datetime.date): Start date for data download.
        end_date (datetime.date): End date for data download.
    Returns:
        pandas.DataFrame: Downloaded stock data.
    """
    df = yf.download(op, start=start_date, end=end_date, progress=False)
    return df

# Sidebar controls for company selection and date range
option = st.sidebar.selectbox("Select Company", ["Google Inc.", "Microsoft", "Tesla", "Air BNB", "Meta"])
option = option.upper()

today = datetime.date.today()
duration = st.sidebar.number_input('Enter the duration (days)', value=3000, min_value=1)
before = today - datetime.timedelta(days=duration)
start_date = st.sidebar.date_input('Start Date', value=before)
end_date = st.sidebar.date_input('End date', today)

# Ensure end_date is not before start_date
if start_date > end_date:
    st.sidebar.error("Error: End date cannot be before start date. Please adjust dates.")
    # Set end_date to start_date to prevent further errors if user doesn't fix it immediately
    end_date = start_date

stock = "" # Initialize stock ticker
if option == "GOOGLE INC.":
    st.sidebar.success(f'Selected: {option}\nStart date: {start_date}\nEnd date: {end_date}')
    stock = "GOOG" # Corrected ticker
elif option == "MICROSOFT":
    st.sidebar.success(f'Selected: {option}\nStart date: {start_date}\nEnd date: {end_date}')
    stock = "MSFT"
elif option == "TESLA":
    st.sidebar.success(f'Selected: {option}\nStart date: {start_date}\nEnd date: {end_date}')
    stock = "TSLA"
elif option == "META":
    st.sidebar.success(f'Selected: {option}\nStart date: {start_date}\nEnd date: {end_date}')
    stock = "META"
elif option == "AIR BNB":
    st.sidebar.success(f'Selected: {option}\nStart date: {start_date}\nEnd date: {end_date}')
    stock = "ABNB"

# Download data based on selected stock and dates
data = pd.DataFrame() # Initialize data as an empty DataFrame
if stock:
    data = download_data(stock, start_date, end_date)
else:
    st.error("Please select a company to display data.")

scaler = StandardScaler()

def tech_indicators():
    """Displays technical indicators for the selected stock."""
    st.header('Technical Indicators')
    if data.empty:
        st.error("No data available for the selected stock and dates. Please adjust the date range or company selection.")
        return
    
    # Check if 'Close' column exists
    if 'Close' not in data.columns:
        st.error("The 'Close' price column is missing from the downloaded data. Cannot calculate indicators.")
        return

    option = st.radio('Choose a Technical Indicator to Visualize', ['Close', 'BB', 'MACD', 'RSI', 'SMA', 'EMA'])

    # Ensure 'Close' column is a 1D Series for ta library functions.
    # .squeeze() will convert a DataFrame with a single column into a Series,
    # or leave a Series unchanged. This helps ensure 1D input.
    close_series = data['Close'].squeeze()

    # Bollinger Bands
    # Use .copy() to avoid modifying the original 'data' DataFrame
    bb = data.copy() 
    bb_indicator = BollingerBands(close_series)
    bb['bb_h'] = bb_indicator.bollinger_hband()
    bb['bb_l'] = bb_indicator.bollinger_lband()
    bb = bb[['Close', 'bb_h', 'bb_l']]

    # MACD
    macd = MACD(close_series).macd()
    # RSI
    rsi = RSIIndicator(close_series).rsi()
    # SMA
    sma = SMAIndicator(close_series, window=14).sma_indicator()
    # EMA
    ema = EMAIndicator(close_series).ema_indicator()

    # Streamlit will use the DataFrame/Series index (which is the Date) as the x-axis
    if option == 'Close':
        st.write('Close Price')
        st.line_chart(data.Close)
    elif option == 'BB':
        st.write('Bollinger Bands')
        st.line_chart(bb)
    elif option == 'MACD':
        st.write('Moving Average Convergence Divergence')
        st.line_chart(macd)
    elif option == 'RSI':
        st.write('Relative Strength Index')
        st.line_chart(rsi)
    elif option == 'SMA':
        st.write('Simple Moving Average')
        st.line_chart(sma)
    else:
        st.write('Exponential Moving Average')
        st.line_chart(ema)

def dataframe():
    """Displays the most recent stock data in a table."""
    st.header('Recent Data')
    if data.empty:
        st.error("No data available to display. Please adjust the date range or company selection.")
    else:
        st.dataframe(data.tail(10))

def predict():
    """Allows users to predict future stock prices using Linear Regression."""
    if data.empty:
        st.error("No data available to predict. Please adjust the date range or company selection.")
        return
    
    if 'Close' not in data.columns:
        st.error("The 'Close' price column is missing from the downloaded data. Cannot make predictions.")
        return

    num = st.number_input('How many days of forecast?', value=1, min_value=1)
    num = int(num)
    if st.button('Predict'):
        engine = LinearRegression()
        model_engine(engine, num)

def model_engine(model, num):
    """
    Trains a Linear Regression model and predicts future stock prices.
    Args:
        model: The machine learning model (e.g., LinearRegression).
        num (int): Number of days to forecast.
    """
    # getting only the closing price
    df = data[['Close']].copy() # Use .copy() to avoid SettingWithCopyWarning

    # Check if df has enough data for shifting and prediction
    if len(df) < num + 1: # Need at least num + 1 days to shift and have data for training
        st.error(f"Not enough data to forecast {num} days. Need at least {num + 1} days of data. Please select a longer duration or fewer forecast days.")
        return

    # shifting the closing price based on number of days forecast
    df['preds'] = df.Close.shift(-num)
    
    # Drop rows with NaN values created by shifting (these are the last 'num' rows)
    df.dropna(inplace=True)

    # After dropping NaNs, ensure there's still data left for training
    if df.empty:
        st.error("Not enough data remaining after preparing for prediction. Adjust date range or forecast days.")
        return

    # Separate features (x) and target (y)
    x = df.drop(['preds'], axis=1).values
    y = df['preds'].values.ravel() 

    # Scale the features
    x = scaler.fit_transform(x)
    
    # Check if there's enough data for forecasting and training
    if len(x) < num:
        st.error(f"Not enough data points ({len(x)}) to create a forecast for {num} days. Increase data duration.")
        return
    
    # Store data for future forecast
    x_forecast = x[-num:]
    
    # Prepare data for model training (remove forecast portion)
    # This is the crucial change: ensure y is also truncated to match x for training
    x_train_val = x[:-num] 
    y_train_val = y[:-num] 

    # Check if x_train_val and y_train_val are empty after preprocessing
    if x_train_val.size == 0 or y_train_val.size == 0:
        st.error("Insufficient data after preprocessing for model training. Adjust date range or forecast days.")
        return

    #spliting the data
    # Ensure there are enough samples for both training and testing after all preprocessing
    if len(x_train_val) < 2 or len(y_train_val) < 2: # Need at least 2 samples for train_test_split
        st.error("Not enough data points to split into training and testing sets. Try a longer duration.")
        return
        
    try:
        x_train, x_test, y_train, y_test = train_test_split(x_train_val, y_train_val, test_size=.2, random_state=7)
    except ValueError as e:
        st.error(f"Error splitting data: {e}. This usually means not enough samples. Try increasing the data duration.")
        return

    # training the model
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    
    st.text(f'Predicted with an accuracy (R2 score) of: {r2_score(y_test, preds):.4f}')
    
    # predicting stock price based on the number of days
    if x_forecast.size == 0:
        st.error("No data available for future forecast. Check 'num' value and data availability.")
        return

    forecast_pred = model.predict(x_forecast).ravel() # Flatten forecast predictions
    st.subheader(f'Forecasted Closing Prices for the next {num} Day(s):')
    for i, pred in enumerate(forecast_pred, start=1):
        st.text(f'Day {i}: {pred:.2f}')

if __name__ == '__main__':
    main()
