# data + analysis functions

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

def get_stock_price(symbol):
    return 1000000.20