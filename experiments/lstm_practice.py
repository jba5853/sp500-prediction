import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# === 1. DOWNLOAD DATA ===
df = yf.download('^GSPC', start='2015-06-01', end='2025-07-01')
data = df[['Close']].copy()
values = data.values

# === 2. SCALE DATA ===
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# === 3. TRAIN/TEST SPLIT ===
n_train = int(len(scaled) * 0.95)

# === 4. BUILD TRAIN SET ===
X_train, y_train = [], []
for i in range(60, n_train):
    X_train.append(scaled[i-60:i, 0])
    y_train.append(scaled[i, 0])
X_train = np.array(X_train).reshape(-1, 60, 1)
y_train = np.array(y_train)

# === 5. BUILD & TRAIN MODEL ===
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(60, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=32, epochs=5, verbose=1)

# === 6. BUILD TEST SET ===
X_test = []
for i in range(n_train, len(scaled)):
    X_test.append(scaled[i-60:i, 0])
X_test = np.array(X_test).reshape(-1, 60, 1)
y_test = values[n_train:]  # raw closes for plotting & accuracy

# === 7. TEST PREDICTIONS ===
pred_scaled = model.predict(X_test)
predictions = scaler.inverse_transform(pred_scaled)

# === 8. DIRECTION ACCURACY – TEST ===
actual_diff = np.diff(y_test.flatten())
pred_diff   = np.diff(predictions.flatten())
test_dir_acc = accuracy_score((actual_diff>0).astype(int),
                              (pred_diff>0).astype(int))

# === 9. DIRECTION ACCURACY – TRAIN ===
train_pred_scaled = model.predict(X_train)
train_preds       = scaler.inverse_transform(train_pred_scaled)
train_actual      = values[60:n_train]
train_diff_actual = np.diff(train_actual.flatten())
train_diff_pred   = np.diff(train_preds.flatten())
train_dir_acc = accuracy_score((train_diff_actual>0).astype(int),
                               (train_diff_pred>0).astype(int))

# === 10. TOMORROW’S PREDICTION ===
last_60 = scaled[-60:].reshape(1, 60, 1)
next_scaled = model.predict(last_60)[0][0]
next_price  = scaler.inverse_transform([[next_scaled]])[0][0]
probability = test_dir_acc * 100
direction   = 'UP' if next_scaled > scaled[-1][0] else 'DOWN'

# === 11. PRINT METRICS ===
print("Model Performance:")
print(f"  Training Direction Accuracy: {train_dir_acc:.2f}")
print(f"  Test     Direction Accuracy: {test_dir_acc:.2f}\n")

print("Prediction for tomorrow:")
print(f"  Predicted Close Price:         ${next_price:.2f}")
print(f"  “Probability” of being right: {probability:.2f}%")
print(f"  Predicted direction:           {direction}")

# === 12. PLOT ACTUAL vs PREDICTED (TEST SET ONLY) ===
plt.figure(figsize=(12,6))
plt.plot(data.index[n_train:], y_test.flatten(),    label='Actual Close')
plt.plot(data.index[n_train:], predictions.flatten(), label='Predicted Close')
plt.title('S&P 500: Actual vs. Predicted Close Price (Test Set)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.tight_layout()
plt.show()
