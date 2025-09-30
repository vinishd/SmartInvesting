import os
import sys
import shutil
from dotenv import load_dotenv
load_dotenv() 
import stock_fetch
import train_lstm
import prepare_dataset  # Import your prepare_dataset.py as a module

def main():
    stock_symbol = input("Enter stock symbol: ").strip().upper()
    if stock_symbol.lower() in ['q', 'quit']:
        print("Exiting the application.")
        sys.exit()

    # Fetch stock data and perform analysis
    stock_fetch.overall_summary(stock_symbol)

    # advanced analysis
    advanced_analysis = input("Do you want to run advanced analysis with LSTM training? (yes/no): ").strip().lower()
    if advanced_analysis in ['yes', 'y']:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
        if os.path.exists(data_dir):
            for filename in os.listdir(data_dir):
                file_path = os.path.join(data_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
        print("Preparing dataset for LSTM analysis...")
        prepare_dataset.prepare_dataset(stock_symbol)
        data_path = f"../data/{stock_symbol}_W60_next1_close.npz"
        scaler_path = "../data/scaler_y.pkl"
        train_lstm.train_lstm_model(data_path, scaler_path)
    else:
        print("Advanced analysis skipped.")

if __name__ == "__main__":
    main()