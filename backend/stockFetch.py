import yfinance as yf
import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
import requests

MY_ENV_VAR = os.getenv('GNEWS_API_KEY')
# print(MY_ENV_VAR)
stock_symbol = input("Enter stock symbol: ")  # Ask for a stock symbol
if stock_symbol.lower() in ['q', 'quit']:
    print("Exit")
    exit()

data = yf.Ticker(stock_symbol)  # Fetch the stock data for the given symbol
print(data)
# news = 

print(f"{stock_symbol} info: {data.info}")  # Prints the information about the given stock symbol
# if data.info["symbol"]:

print("\n")

# print("MSFT calendar", data.calendar)  # Prints the calendar for Microsoft
# print("\n")

# print("Analyst price targets for Microsoft", data.analyst_price_targets)  # Prints the analyst price targets for Microsoft
# print("\n")

# print("Quarterly income statement", data.quarterly_income_stmt)  # Prints the quarterly income statement for Microsoft
# print("\n")

# print("Last month of historical data", data.history(period='1mo'))  # Prints the last month of historical data for Microsoft
# print("\n")

# print("Call Options for the nearest expiration date\n", data.option_chain(data.options[0]).calls)  # Prints the call options for the nearest expiration date
# print("\n")

# print(dat.history(period="1mo"))  # Fetches the last month of historical data for Microsoft
