import yfinance as yf
import os
import json
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
import requests
import finnhub
from newspaper import Article
from newsapi import NewsApiClient
from google.cloud import language_v1
# GNEWS_API_KEY = os.getenv('GNEWS_API_KEY')
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
google_client = language_v1.LanguageServiceClient()
# newsapi = NewsApiClient(api_key=GNEWS_API_KEY)
# /v2/top-headlines


# print(MY_ENV_VAR)
stock_symbol = input("Enter stock symbol: ").strip().upper()  # Ask for a stock symbol
if stock_symbol.lower() in ['q', 'quit']:
    print("Exit")
    exit()
data = yf.Ticker(stock_symbol)

# https://api.marketaux.com/v1/entity/stats/aggregation?symbols=TSLA,AMZN,MSFT&published_after=2025-03-22T08:21&language=en&api_token={key}}

# https://newsapi.org/v2/top-headlines?q=appl&language=en&from=2025-02-24&to=2025-03-21&sortBy=popularity&apiKey={GNEWS_API_KEY}


# news and recommendations [ works ]
# finnhub_client = finnhub.Client(api_key={FINNHUB_API_KEY})
response = (finnhub_client.company_news(
            stock_symbol, 
            _from="2025-02-25",
            to="2025-03-23",
            )
            )

table1  = [[] for _ in range(51)]
for index, news_item in enumerate(response[:50]):  # Limit to the most recent 50 news items
    # print(f"News item {index + 1}:")
    # print(json.dumps(news_item, indent=4))
    try:
        article = Article(news_item['url'])
        article.download()
        article.parse()
        text = article.text
    except:
        text = ""
    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
    analysis_response = google_client.analyze_sentiment(document=document)
    score = analysis_response.document_sentiment.score # overall sentiment: -1 = negative, 0 = neutral, 1 = positive
    magnitude = analysis_response.document_sentiment.magnitude # strength of emotion: 0 = no emotion, higher values = stronger emotion
    table1[index].append(news_item['headline'])
    table1[index].append(score)
    table1[index].append(magnitude)
    table1[index].append(news_item['datetime'])
    # print("\n")

print("Sentiment Analysis of News Articles:")
for i in range(len(table1) - 1):
    print(f"News item {i + 1}:")
    print(f"Headline: {table1[i][0]}")
    print(f"Sentiment Score: {table1[i][1]}")
    print(f"Sentiment Magnitude: {table1[i][2]}")
    # print(f"Datetime: {table1[i][3]}")
    print("\n")

recommendations = finnhub_client.recommendation_trends(stock_symbol)
total_score = 0
all_recommendation_counts = 0
for index, rec in enumerate(recommendations):
    score = (
        2 * rec['strongBuy'] +
        1 * rec['buy'] +
        0 * rec['hold'] -
        1 * rec['sell'] -
        2 * rec['strongSell']
    )
    total_score += score
    all_recommendation_counts += (
        rec['strongBuy'] + rec['buy'] + rec['hold'] + rec['sell'] + rec['strongSell']
    )
    # print(f"Index {index + 1}:")
    # print(json.dumps(rec, indent=4))
    # print("\n")

normalized_score = total_score / all_recommendation_counts
if normalized_score > 1:
    sentiment_label = "Strong Buy"
elif normalized_score > 0.5:
    sentiment_label = "Buy"
elif normalized_score > -0.5:
    sentiment_label = "Neutral"
elif normalized_score > -1:
    sentiment_label = "Sell"
else:
    sentiment_label = "Strong Sell"

print(f"📊 Analyst Consensus Score: {normalized_score:.2f}. This means the latest recommendation for {stock_symbol} is: {sentiment_label}")






# price and trends
current_price_of_stock = finnhub_client.quote(stock_symbol)['c']
previous_close_price = finnhub_client.quote(stock_symbol)['pc']
trend = current_price_of_stock - previous_close_price

price_30_days_ago = data.history(period='1mo').iloc[0]['Close']  # Closing price from 30 days ago
trend_pct = (current_price_of_stock - price_30_days_ago) / price_30_days_ago * 100
# print(f"{stock_symbol} has changed by {trend_pct:.2f}% over the last 30 days")

if trend > 0:
    if trend_pct < 0:
        print(f"{stock_symbol} is trending upwards by {abs(trend):.2f} points in the past day and has decreased by {abs(trend_pct):.2f}% over the last 30 days")
    else:
        print(f"{stock_symbol} is trending upwards by {abs(trend):.2f} points in the past day and has increased by {trend_pct:.2f}% over the last 30 days")
elif trend < 0:
    if trend_pct < 0:
        print(f"{stock_symbol} is trending downwards by {abs(trend):.2f} points in the past day and has decreased by {abs(trend_pct):.2f}% over the last 30 days")
    else:
        print(f"{stock_symbol} is trending downwards by {abs(trend):.2f} points in the past day and has increased by {trend_pct:.2f}% over the last 30 days")
else:
    print(f"{stock_symbol} is stable with no change in price")


 # Fetch the stock data for the given symbol
# for i in data.history(period="1mo").index:
#     print(f"Date: {i.date()}, Close Price: {data.history(period='1mo').loc[i]['Close']}")


# print(f"{stock_symbol} info: {data.info}")  # Prints the information about the given stock symbol

# print("\n")

# print("APPL calendar", data.get_info)  # Prints the calendar for Microsoft
# print("\n")
# yf.Ticker("MSFT").analyst_price_targets
# print("Analyst price targets for AAPL", data.analyst_price_targets)  # Prints the analyst price targets for Microsoft
# print("\n")
# news = yf.Search("AAPL", news_count=10).news
# print("NEWS : \n")
# print(json.dumps(news, indent=4))  # Print the news related to the stock symbol
# print("Quarterly income statement", data.quarterly_income_stmt)  # Prints the quarterly income statement for Microsoft
# print("\n")

# print("Last month of historical data", data.history(period='1mo'))  # Prints the last month of historical data for Microsoft
# print("\n")

# print("Call Options for the nearest expiration date\n", data.option_chain(data.options[0]).calls)  # Prints the call options for the nearest expiration date
# print("\n")

# print(dat.history(period="1mo"))  # Fetches the last month of historical data for Microsoft
