# Smart Investing App

## Overview
The Smart Investing App is a Python-based application designed to provide users with insights into stock performance through data fetching and advanced analysis using LSTM (Long Short-Term Memory) models. The application integrates stock data retrieval from various APIs and employs machine learning techniques to predict stock prices.

## Project Structure
```
smart-investing-app
├── src
│   ├── main.py               # Entry point of the application
│   ├── stock_fetch.py        # Functions to fetch stock data and perform analysis
│   ├── train_lstm.py         # LSTM model training and evaluation
│   └── utils
│       └── __init__.py       # Utility functions and classes
├── data
│   └── (place your data files here)  # Directory for data files
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Setup Instructions
1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd smart-investing-app
   ```

2. **Create a virtual environment** (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies**:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. **Run the application**:
   ```
   python src/main.py
   ```

2. **Follow the prompts** to enter a stock symbol. The application will fetch stock data and provide options for advanced analysis, including LSTM training.

## Dependencies
- yfinance
- finnhub
- tensorflow
- numpy
- pandas
- matplotlib
- scikit-learn
- newspaper3k
- google-cloud-language

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.