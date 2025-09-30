# SmartInvesting

### Features
Real-time financial data: Stock/crypto prices, market cap, volume, etc.

News & sentiment analysis: Scrape or fetch headlines and run NLP to detect positive/negative sentiment.

Buy/Sell Signals: Use ML models (e.g., time series + sentiment) to generate basic investment signals. Integrated percentage buy & sell



### Tech Stack
## Backend/Data:

Google Cloud BigQuery – store historical data for analysis

Vertex AI – model training (e.g., LSTM for forecasting, sentiment classifiers)

Cloud Functions – automate fetching data

Yahoo Finance API – for financial market data

## NLP/Sentiment:
Cloud Natural Language API or Vertex AI – run sentiment analysis on news

(future plan) integrate LangChain + Gemini/ChatGPT to summarize insights


## To run:
- Clone Repo
- Generate and import FINNHUB & YFINANCE keys
- navigate to /src/main.py
- run python3 main.py