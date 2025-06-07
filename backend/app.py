from flask import Flask, jsonify, request, abort
import re
from flask_cors import CORS
from analysis_utils import get_stock_price

# Strictly FLASK APP


app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])  # Explicitly allow your React app

@app.route('/api/data')
def get_data():
    print("API endpoint hit!")  # Debug print
    return jsonify({"message": "Hello!"})

@app.route('/')
def home():
    return "Flask server is running!"

@app.route('/api/price')
def get_price():
    symbol = request.args.get('symbol')
    if not symbol or not re.fullmatch(r'[A-Z]{1,5}', symbol.upper()):
        return jsonify({'error': 'Invalid symbol'}), 400

    price = get_stock_price(symbol)
    return jsonify({'symbol': symbol, 'price': price})

if __name__ == '__main__':
    print("Starting Flask server on port 5000...")
    app.run(debug=True, port=5000)