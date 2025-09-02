import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
import os


# datapath stuff to get the right file(s) :(
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "..", "data")
data_path = os.path.join(data_dir, "AAPL_W60_next1_close.npz")
scaler_path = os.path.join(data_dir, "scaler_y.pkl")


# loading data
data = np.load(data_path)
X_train, Y_train = data['X_train'], data['Y_train']
X_val, Y_val = data['X_val'], data['Y_val']
X_test, Y_test = data['X_test'], data['Y_test']
scaler_y = joblib.load(scaler_path)

#building/training the model... 
model = Sequential([LSTM(50, return_sequences=True, input_shape = (X_train.shape[1], X_train.shape[2])), LSTM(50), Dense(1)])

model.compile(optimizer='adam', loss='mse')

# ------------------------------------------
# Do we dare double stack the LSTMs?
# model = Sequential([
#     LSTM(50, return_sequences=True, input_shape = (X_train.shape[1], X_train.shape[2])), # outputs the full sequence for the next LSTM
#     Dropout(0.2), # avoid overfitting
#     LSTM(50), # last hidden state for Dense
#     Dropout(0.2), # overfitting ^
#     Dense(1)])
# ------------------------------------------


# ---------------------------
# # 100 epochs? Why not..
# history = model.fit(
#     X_train, Y_train, validation_data = (X_val, Y_val),
#     epochs=100, batch_size=32
# )
# ---------------------------

history = model.fit(
    X_train, Y_train, validation_data = (X_val, Y_val),
    epochs=50, batch_size=32
)


# eval
testLoss = model.evaluate(X_test, Y_test)
print("Test MSE:", testLoss)
print("Test RMSE:", np.sqrt(testLoss))

# pred + plot
ypred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(ypred_scaled)
y_test_real = scaler_y.inverse_transform(Y_test.reshape(-1,1))

# pred vs actual
plt.figure(figsize=(10,4))
plt.plot(y_test_real[:100], label="Actual")
plt.plot(y_pred[:100], label="Predicted")
plt.legend()
plt.show()
