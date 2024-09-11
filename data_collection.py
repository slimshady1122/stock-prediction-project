import yfinance as yf

# Define the stock ticker
ticker_symbol = 'AAPL'   # Apple

# Download the historical data (last 5 years of daily data)
stock_data = yf.download(ticker_symbol, start='2018-01-01', end='2023-01-01')

# Display the first few rows
print(stock_data.head())


stock_data.to_csv('apple_stock_data.csv')

