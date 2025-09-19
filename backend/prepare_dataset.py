import numpy as np # type: ignore
import pandas as pd # type: ignore
import yfinance as yf # type: ignore
from sklearn.preprocessing import StandardScaler, MinMaxScaler # type: ignore
import os
import joblib # type: ignore

# Initial Ticker and Dates
TICKER = "TSLA"
START, END = "2020-09-01", "2025-08-31"
WINDOW, HORIZON = 60, 1

# Pre-Processing
df = yf.download(TICKER, start=START, end=END, progress=False)
series = df["Close"].dropna()

# windowing (predicting next day's price)
def make_windows(S, window=WINDOW, horizon=HORIZON):
    X, Y = [], []
    values = S.values
    for i in range(len(values) - window - horizon + 1):
        X.append(values[i:i+window])
        Y.append(values[i+window:i+window+horizon][0])
    X = np.array(X)[..., np.newaxis]
    Y = np.array(Y)
    return X, Y

X, Y = make_windows(series)

# Split by Time (70% Train, 15% Val, 15% Test) (based on Windows, not raw)
n = len(X)
i_train, i_val = int(n * 0.7), int(n * 0.85)
X_train, Y_train = X[:i_train], Y[:i_train]
X_val, Y_val = X[i_train:i_val], Y[i_train:i_val]
X_test, Y_test = X[i_val:], Y[i_val:]


# Fit StdScalar on Train data (X AND Y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
X_val_scaled   = scaler.transform(X_val.reshape(-1, 1)).reshape(X_val.shape)
X_test_scaled  = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

# Y
scaler_y = MinMaxScaler()
Y_train_scaled = scaler_y.fit_transform(Y_train.reshape(-1, 1)).ravel()
Y_val_scaled   = scaler_y.transform(Y_val.reshape(-1, 1)).ravel()
Y_test_scaled  = scaler_y.transform(Y_test.reshape(-1, 1)).ravel()

# send to backend/data/...
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "..", "data")
os.makedirs(data_dir, exist_ok=True)

data_path = os.path.join(data_dir, "TSLA_W60_next1_close.npz")
np.savez(data_path,
         X_train=X_train_scaled, Y_train=Y_train_scaled,
         X_val=X_val_scaled,     Y_val=Y_val_scaled,
         X_test=X_test_scaled,   Y_test=Y_test_scaled)
scaler_y_path = os.path.join(data_dir, "scaler_y.pkl")
joblib.dump(scaler_y, scaler_y_path)

print("Shapes:")
print("Train:", X_train.shape, Y_train.shape)
print("Val:  ", X_val.shape, Y_val.shape)
print("Test: ", X_test.shape, Y_test.shape)