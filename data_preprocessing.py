import pandas as pd

# Load the data
stock_data = pd.read_csv('apple_stock_data.csv', index_col='Date', parse_dates=True)

# Strip any leading/trailing spaces from column names
stock_data.columns = stock_data.columns.str.strip()

# Print column names for debugging
print(stock_data.columns)

# Handling missing values
stock_data = stock_data.dropna()

# Adding technical indicators
# Simple Moving Average (SMA)
stock_data['SMA_20'] = stock_data['Close'].rolling(window=20).mean()

# Exponential Moving Average (EMA)
stock_data['EMA_20'] = stock_data['Close'].ewm(span=20, adjust=False).mean()

# Relative Strength Index (RSI)
def calculate_rsi(data, window):
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

stock_data['RSI_14'] = calculate_rsi(stock_data, 14)

# Optional: Save the updated data
stock_data.to_csv('apple_stock_data_with_indicators.csv')

# Example to check and add missing columns
if 'Price_Change' not in stock_data.columns:
    stock_data['Price_Change'] = stock_data['Close'].pct_change()

if 'Volume_Change' not in stock_data.columns:
    stock_data['Volume_Change'] = stock_data['Volume'].pct_change()

# Save the processed data again
stock_data.to_csv('apple_stock_data_with_indicators.csv')

# Display the updated dataset
print(stock_data.tail())
