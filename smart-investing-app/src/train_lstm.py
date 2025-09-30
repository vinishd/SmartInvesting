import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from sklearn.preprocessing import MinMaxScaler  # type: ignore
import joblib  # type: ignore
import os
import pandas as pd  # type: ignore

def train_lstm_model(data_path, scaler_path, epochs=50, batch_size=32):
    # loading data
    data = np.load(data_path)
    X_train, Y_train = data['X_train'], data['Y_train']
    X_val, Y_val = data['X_val'], data['Y_val']
    X_test, Y_test = data['X_test'], data['Y_test']
    scaler_y = joblib.load(scaler_path)

    # building/training the model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(50),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    history = model.fit(
        X_train, Y_train, validation_data=(X_val, Y_val),
        epochs=epochs, batch_size=batch_size
    )

    # eval
    testLoss = model.evaluate(X_test, Y_test)
    print("Test MSE:", testLoss)
    print("Test RMSE:", np.sqrt(testLoss))

    # pred + plot
    ypred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(ypred_scaled)
    y_test_real = scaler_y.inverse_transform(Y_test.reshape(-1, 1))

    # pred vs actual
    plt.figure(figsize=(10, 4))
    plt.plot(y_test_real[:100], label="Actual")
    plt.plot(y_pred[:100], label="Predicted")
    plt.legend()
    plt.show()

    # backtest time
    backtest_df = pd.DataFrame({
        'actual_price': y_test_real.flatten(),
        'predicted_price': y_pred.flatten()
    })

    # Calc actual returns
    backtest_df['actual_return'] = backtest_df['actual_price'].pct_change()

    # calc predicted price change
    backtest_df['actual_today'] = backtest_df['actual_price'].shift(1)
    backtest_df['predicted_change'] = (backtest_df['predicted_price'] - backtest_df['actual_today']) / backtest_df['actual_today']

    # check if signals are inverted
    temp_signals = np.where(backtest_df['predicted_change'] > 0.001, 1,
                             np.where(backtest_df['predicted_change'] < -0.001, -1, 0))

    buy_mask = (temp_signals == 1) & (~np.isnan(backtest_df['actual_return'].shift(-1)))
    sell_mask = (temp_signals == -1) & (~np.isnan(backtest_df['actual_return'].shift(-1)))

    if len(buy_mask) > 0 and len(sell_mask) > 0:
        avg_buy_result = backtest_df['actual_return'].shift(-1)[buy_mask].mean()
        avg_sell_result = backtest_df['actual_return'].shift(-1)[sell_mask].mean()

        print(f"after BUY signals, avg return: {avg_buy_result:.4f}")
        print(f"after SELL signals, avg return: {avg_sell_result:.4f}")

        flip_signals = avg_buy_result < 0 and avg_sell_result > 0
    else:
        flip_signals = False

    # signal gen
    tau = 0.001
    if flip_signals:
        backtest_df['signal'] = np.where(backtest_df['predicted_change'] > tau, -1,
                                          np.where(backtest_df['predicted_change'] < -tau, 1, 0))
    else:
        backtest_df['signal'] = np.where(backtest_df['predicted_change'] > tau, 1,
                                          np.where(backtest_df['predicted_change'] < -tau, -1, 0))

    backtest_df['position'] = backtest_df['signal'].shift(1).fillna(0)

    # strategy returns
    backtest_df['strategy_return'] = backtest_df['position'] * backtest_df['actual_return']

    # Cumulative performance
    initial = 10000
    backtest_df['strategy_equity'] = initial * (1 + backtest_df['strategy_return']).cumprod()
    backtest_df['buyhold_equity'] = initial * (1 + backtest_df['actual_return']).cumprod()

    # analysis
    print("\n=== TRADING ANALYSIS ===")
    print(f"Total trades: {(backtest_df['signal'].abs() > 0).sum()}")
    print(f"Long signals: {(backtest_df['signal'] == 1).sum()}")
    print(f"Short signals: {(backtest_df['signal'] == -1).sum()}")

    # plot
    plt.figure(figsize=(12, 6))
    plt.plot(backtest_df['strategy_equity'], label='LSTM Strategy', linewidth=2)
    plt.plot(backtest_df['buyhold_equity'], label='Buy & Hold', linewidth=2)
    plt.legend()
    plt.title('LSTM Trading Strategy vs Buy & Hold')
    plt.grid(True, alpha=0.3)
    plt.show()

    # final results
    final_strategy = backtest_df['strategy_equity'].iloc[-1]
    final_buyhold = backtest_df['buyhold_equity'].iloc[-1]

    print(f"\n=== RESULTS ===")
    print(f"Strategy: ${final_strategy:,.2f} ({(final_strategy/initial-1)*100:.1f}%)")
    print(f"Buy & Hold: ${final_buyhold:,.2f} ({(final_buyhold/initial-1)*100:.1f}%)")

    # Performance metrics
    strategy_returns = backtest_df['strategy_return'].dropna()
    buyhold_returns = backtest_df['actual_return'].dropna()

    print(f"\n=== PERFORMANCE METRICS ===")
    print(f"Strategy Sharpe: {strategy_returns.mean() / strategy_returns.std() * np.sqrt(252):.2f}")
    print(f"Buy & Hold Sharpe: {buyhold_returns.mean() / buyhold_returns.std() * np.sqrt(252):.2f}")
    print(f"Strategy Win Rate: {(strategy_returns > 0).mean() * 100:.1f}%")

    # trading signals gen
    backtest_df['signal'] = temp_signals

    # next action
    last_signal = backtest_df['signal'].iloc[-1]
    if last_signal == 1:
        suggestion = "BUY"
    elif last_signal == -1:
        suggestion = "SELL"
    else:
        suggestion = "HOLD"
    print(f"\nLSTM SUGGESTION for the next trading day: {suggestion}")

    x = 100

    # Use the last predicted change as confidence
    last_pred_change = backtest_df['predicted_change'].iloc[-1]

    # scale sell/hold suggestion based on confidence
    if last_signal == -1:
        # stronger negative = sell more
        sell_fraction = min(max(abs(last_pred_change) * 2, 0.1), 1.0)  # between 10% and 100%
        shares_to_sell = int(x * sell_fraction)
        print(f"\nSuggested action: SELL {shares_to_sell} out of {x} shares ({sell_fraction*100:.1f}%)")
        print(f"Hold the remaining {x - shares_to_sell} shares.")
    elif last_signal == 1:
        print(f"\nSuggested action: HOLD or consider buying more (model predicts increase).")
    else:
        print(f"\nSuggested action: HOLD all {x} shares (no strong signal).")

    
    print("\nRecent LSTM signals (most recent last):")
    signal_map = {1: "BUY", -1: "SELL", 0: "HOLD"}
    print([signal_map.get(sig, "HOLD") for sig in backtest_df['signal'].tail(5)])