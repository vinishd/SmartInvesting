import yfinance as yf
import argparse
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
recommendations = finnhub_client.recommendation_trends(stock_symbol)
current_price_of_stock = finnhub_client.quote(stock_symbol)['c'] # Current price
previous_close_price = finnhub_client.quote(stock_symbol)['pc'] # Previous close price
price_30_days_ago = data.history(period='1mo').iloc[0]['Close']  # Closing price from 30 days ago


# news
response = (finnhub_client.company_news(
            stock_symbol, 
            _from="2025-02-25",
            to="2025-03-23",
            )
            )

def compute_sentiment_score(table1):
    """
    Compute the sentiment score from the table of news articles.
    """
    print("Computing sentiment score from news articles...")
    table1  = [[] for _ in range(51)]
    for index, news_item in enumerate(response[:50]):  # Limit to the most recent 50 news items

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
    avg_sentiment = sum([row[1] for row in table1 if row]) / len([row for row in table1 if row])
    
    if avg_sentiment > 0.3:
        sentiment_label = "favorable sentiment"
    elif avg_sentiment < -0.3:
        sentiment_label = "disadvantageous sentiment"
    else:
        sentiment_label = "neutral sentiment"
    return avg_sentiment, sentiment_label




# recommendations
def calculate_consensus_score(recommendations):
    """
    Calculate the average consensus score based on analysis of recommendations.
    """
    print("Calculating consensus score based on expert recommendations...")
    total_score = 0
    all_recommendation_counts = 0
    for _, rec in enumerate(recommendations):
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

    
    normalized_score = total_score / all_recommendation_counts
    if normalized_score > 1:
        sentiment_label = "Strongly Buy"
    elif normalized_score > 0.5:
        sentiment_label = "Buy"
    elif normalized_score > -0.5:
        sentiment_label = "Hold"
    elif normalized_score > -1:
        sentiment_label = "Sell"
    else:
        sentiment_label = "Strongly Sell"

    return normalized_score, sentiment_label


# price and trends
def calc_trend(current_price_of_stock, previous_close_price, price_30_days_ago):
    
    """
    Calculate the trend percentage based on current price, previous close and price 30 days ago.
    """
    print("Calculating trend analysis...")
    trend_pct = 0
    short_term_trend = current_price_of_stock - previous_close_price  # Short term trend (1 day)
    if price_30_days_ago:
        trend_pct = (current_price_of_stock - price_30_days_ago) / price_30_days_ago * 100
        
    if short_term_trend > 0:
        short_term_trend_label = f"trending upwards by {abs(short_term_trend):.2f} points"
    elif short_term_trend < 0:
        short_term_trend_label = f"trending downwards by {abs(short_term_trend):.2f} points"

    if trend_pct < 0:
        trend_pct_label = f"decreased by {abs(trend_pct):.2f}%"
    elif trend_pct > 0:
        trend_pct_label = f"increased by {trend_pct:.2f}%"
    else:
        trend_pct_label = "no change"
    
    return short_term_trend_label, trend_pct_label


def short_summary():
    """
    Print a short summary of stock analysis.
    """
    _, sentiment_label = calculate_consensus_score(recommendations)
    print("Based on the latest analyst trends for this company, it's recommended to "
          f"{sentiment_label} {stock_symbol}\n")

def overall_summary():
    """
    Print the overall summary of stock analysis.
    """
    short_term_trend_label, trend_pct_label = calc_trend(current_price_of_stock, previous_close_price, price_30_days_ago)
    normalized_score, recommendation_sentiment_label = calculate_consensus_score(recommendations) # Calculate the consensus score from analyst recommendations
    news_sentiment_score, news_sentiment_label = compute_sentiment_score(response)  # Compute the sentiment score from news articles

    


    print(f"\n📈 Current Price of {stock_symbol}: ${current_price_of_stock:.2f}. It last closed at ${previous_close_price}")

    print(f"In the past day, {stock_symbol} has been {short_term_trend_label} and has {trend_pct_label} over the last 30 days.")

    print(f"📊 The average consensus score based on the latest analyst trends for this company is {normalized_score:.2f} "
          f"therefore the expert recommendation for {stock_symbol} is to {recommendation_sentiment_label}.")

    print(f"📰 Recent media coverage (over the past month) reflects {news_sentiment_label} towards {stock_symbol} (News Sentiment Score: {news_sentiment_score:.2f})")

overall_summary()


# if __name__ == "__main__":

#     parser = argparse.ArgumentParser(description="Smart Investment Insights CLI")
#     parser.add_argument("symbol", help="Stock symbol to analyze (e.g. AAPL)")
#     parser.add_argument("--summary", help="Show only the final insight summary")

#     args = parser.parse_args()
#     stock_symbol = args.symbol.upper()
#     summary_only = args.summary

#     if summary_only:
#         print("📌 Showing summary only...")

#     else:
#         print(f"\n📈 Analyzing stock data for {stock_symbol}...\n")
#         overall_summary()


# trend_pct = (current_price_of_stock - price_30_days_ago) / price_30_days_ago * 100
# print(f"{stock_symbol} has changed by {trend_pct:.2f}% over the last 30 days")

# if trend > 0:
#     if trend_pct < 0:
#         print(f"{stock_symbol} is trending upwards by {abs(trend):.2f} points in the past day and has decreased by {abs(trend_pct):.2f}% over the last 30 days")
#     else:
#         print(f"{stock_symbol} is trending upwards by {abs(trend):.2f} points in the past day and has increased by {trend_pct:.2f}% over the last 30 days")
# elif trend < 0:
#     if trend_pct < 0:
#         print(f"{stock_symbol} is trending downwards by {abs(trend):.2f} points in the past day and has decreased by {abs(trend_pct):.2f}% over the last 30 days")
#     else:
#         print(f"{stock_symbol} is trending downwards by {abs(trend):.2f} points in the past day and has increased by {trend_pct:.2f}% over the last 30 days")
# else:
#     print(f"{stock_symbol} is stable with no change in price")


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
 