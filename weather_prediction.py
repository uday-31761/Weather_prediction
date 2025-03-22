import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, SimpleRNN
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load dataset
df = pd.read_csv("weather.csv")

# Convert date column to datetime
df["Date.Full"] = pd.to_datetime(df["Date.Full"])
df = df.sort_values(by="Date.Full")

# Encode categorical variables
categorical_cols = ["Station.City", "Station.Code", "Station.Location", "Station.State"]
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# Normalize numerical features
scaler = MinMaxScaler()
numerical_cols = [
    "Data.Precipitation", "Data.Temperature.Avg Temp", "Data.Temperature.Max Temp",
    "Data.Temperature.Min Temp", "Data.Wind.Direction", "Data.Wind.Speed"
]
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Feature Correlation Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.close()

# Yearly Climate Trends by City
df["Year"] = df["Date.Full"].dt.year
city_trends = df.groupby(["Year", "Station.City"])["Data.Temperature.Avg Temp"].mean().unstack()
city_trends.plot(marker="o", colormap="viridis", figsize=(12, 6))
plt.title("Yearly Temperature Trends by City")
plt.xlabel("Year")
plt.ylabel("Average Temperature")
plt.legend(title="City", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.savefig("climate_trends.png")
plt.close()

# Prepare dataset for time-series forecasting
SEQ_LENGTH = 10
features = df.drop(columns=["Data.Temperature.Avg Temp", "Date.Full", "Year"]).values
target = df["Data.Temperature.Avg Temp"].values
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)
train_generator = TimeseriesGenerator(X_train, y_train, length=SEQ_LENGTH, batch_size=32)
test_generator = TimeseriesGenerator(X_test, y_test, length=SEQ_LENGTH, batch_size=32)

# Build RNN Model
rnn_model = Sequential([
    SimpleRNN(64, activation='relu', return_sequences=True, input_shape=(SEQ_LENGTH, X_train.shape[1])),
    Dropout(0.2),
    SimpleRNN(32, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

rnn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
rnn_model.summary()

# Train RNN Model
rnn_history = rnn_model.fit(train_generator, epochs=30, validation_data=test_generator)

# Save RNN Model
rnn_model.save("climate_rnn_model.h5")

# Build LSTM Model
lstm_model = Sequential([
    LSTM(64, activation='relu', return_sequences=True, input_shape=(SEQ_LENGTH, X_train.shape[1])),
    Dropout(0.2),
    LSTM(32, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
lstm_model.summary()

# Train LSTM Model
lstm_history = lstm_model.fit(train_generator, epochs=30, validation_data=test_generator)

# Plot Loss and Accuracy
plt.figure(figsize=(10, 5))
plt.plot(lstm_history.history['loss'], label='LSTM Train Loss')
plt.plot(lstm_history.history['val_loss'], label='LSTM Validation Loss')
plt.plot(rnn_history.history['loss'], label='RNN Train Loss', linestyle='dashed')
plt.plot(rnn_history.history['val_loss'], label='RNN Validation Loss', linestyle='dashed')
plt.legend()
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig("training_loss_comparison.png")
plt.close()

# Save LSTM Model
lstm_model.save("climate_lstm_model.h5")
print("Model training complete and saved as 'climate_lstm_model.h5' and 'climate_rnn_model.h5'.")

# Prediction Function
def predict_temperature(location, date):
    date = pd.to_datetime(date)
    encoded_location = LabelEncoder().fit_transform([location])[0]
    input_data = np.array([[encoded_location, date.year, date.month, date.day]])
    input_data = scaler.transform(input_data)
    input_data = input_data.reshape((1, SEQ_LENGTH, input_data.shape[1]))
    prediction = lstm_model.predict(input_data)
    return scaler.inverse_transform(prediction.reshape(-1, 1))[0][0]

print("Prediction function ready. Call predict_temperature(location, date) to get temperature forecasts.")