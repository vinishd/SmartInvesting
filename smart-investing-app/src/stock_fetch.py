import yfinance as yf
import os
from dotenv import load_dotenv
import finnhub
from google.cloud import language_v1
import pandas as pd
from newspaper import Article
import numpy as np

load_dotenv()
FINNHUB_API_KEY = os.getenv('FINNHUB_KEY')
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
google_client = language_v1.LanguageServiceClient()

def fetch_stock_data(stock_symbol):
    print(f"Fetching data for {stock_symbol}...")
    data = yf.Ticker(stock_symbol)
    recommendations = finnhub_client.recommendation_trends(stock_symbol)
    current_price_of_stock = finnhub_client.quote(stock_symbol)['c']
    previous_close_price = finnhub_client.quote(stock_symbol)['pc']
    price_30_days_ago = data.history(period='1mo').iloc[0]['Close']
    print("Stock data fetched.")
    return data, recommendations, current_price_of_stock, previous_close_price, price_30_days_ago

def compute_sentiment_score(news_items):
    print("Analyzing news sentiment. This may take a minute. Please wait...")
    table1 = [[] for _ in range(51)]
    for index, news_item in enumerate(news_items[:50]):
        try:
            article = Article(news_item['url'])
            article.download()
            article.parse()
            text = article.text
        except:
            text = ""

        document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
        analysis_response = google_client.analyze_sentiment(document=document)
        score = analysis_response.document_sentiment.score
        magnitude = analysis_response.document_sentiment.magnitude

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
    
    print("News sentiment analysis complete.")
    return avg_sentiment, sentiment_label

def calculate_consensus_score(recommendations):
    print("Calculating analyst consensus score...")
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

    print("Analyst consensus score calculated.")
    return normalized_score, sentiment_label

def calc_trend(current_price_of_stock, previous_close_price, price_30_days_ago):
    print("Calculating price trends...")
    trend_pct = 0
    short_term_trend = current_price_of_stock - previous_close_price
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
    
    print("Price trend calculation complete.")
    return short_term_trend_label, trend_pct_label

def overall_summary(stock_symbol):
    print("\n=== Starting Stock Analysis ===")
    data, recommendations, current_price_of_stock, previous_close_price, price_30_days_ago = fetch_stock_data(stock_symbol)
    short_term_trend_label, trend_pct_label = calc_trend(current_price_of_stock, previous_close_price, price_30_days_ago)
    normalized_score, recommendation_sentiment_label = calculate_consensus_score(recommendations)
    
    print("Fetching recent news for sentiment analysis...")
    news = finnhub_client.company_news(stock_symbol, _from="2025-02-25", to="2025-03-23")
    news_sentiment_score, news_sentiment_label = compute_sentiment_score(news)

    # Determine conjunction ("and" or "but") for trend summary
    short_term_positive = "upwards" in short_term_trend_label
    long_term_positive = "increased" in trend_pct_label

    if short_term_positive == long_term_positive:
        conjunction = "and"
    else:
        conjunction = "but"

    print("\n=== Stock Analysis Complete ===")
    print(f"\n📈 Current Price of {stock_symbol}: ${current_price_of_stock:.2f}. \n It last closed at ${previous_close_price}")
    print(f"In the past 24 hours, {stock_symbol} has been {short_term_trend_label} {conjunction} has {trend_pct_label} over the last 30 days.")
    print(f"📊 The average consensus score based on the latest analyst trends for this company is {normalized_score:.2f} "
          f", therefore the expert recommendation for {stock_symbol} is to {recommendation_sentiment_label}.")
    print(f"📰 Recent media coverage reflects {news_sentiment_label} towards {stock_symbol} (News Sentiment Score: {news_sentiment_score:.2f})")