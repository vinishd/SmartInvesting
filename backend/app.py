from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/api/data')
def index():
    return jsonify({"message": "Welcome to the Smart Investment Insights API!"})







if __name__ == '__main__':
    app.run(port=5000)