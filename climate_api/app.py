import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os

app = Flask(__name__)

df = pd.read_csv("weather.csv")
df["Date.Full"] = pd.to_datetime(df["Date.Full"])

model_path_lstm = "models/climate_lstm_model.h5"
model_path_rnn = "models/climate_rnn_model.h5"

if not os.path.exists(model_path_lstm):
    print(f"Error: LSTM model not found at {model_path_lstm}")
    exit()

if not os.path.exists(model_path_rnn):
    print(f"Error: RNN model not found at {model_path_rnn}")
    exit()

try:
    lstm_model = tf.keras.models.load_model(model_path_lstm)
    rnn_model = tf.keras.models.load_model(model_path_rnn)
except Exception as e:
    print(f"Error loading models: {e}")
    print(f"Tensorflow version: {tf.__version__}")
    exit()

categorical_cols = ["Station.City", "Station.Code", "Station.Location", "Station.State"]
encoders = {col: LabelEncoder().fit(df[col]) for col in categorical_cols}

scaler = MinMaxScaler()
numerical_cols = [
    "Data.Precipitation", "Data.Temperature.Max Temp",
    "Data.Temperature.Min Temp", "Data.Wind.Direction", "Data.Wind.Speed"
]
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

SEQ_LENGTH = 10

def predict_temperature(date, city, model_type="lstm"):
    try:
        date = pd.to_datetime(date)
        year, month, day = date.year, date.month, date.day
        encoded_city = encoders["Station.City"].transform([city])[0]

        input_data = np.array([[year, month, day, encoded_city]])
        input_data = scaler.transform(input_data)
        input_data = input_data.reshape((1, SEQ_LENGTH, input_data.shape[1]))

        if model_type == "lstm":
            prediction = lstm_model.predict(input_data)
        elif model_type == "rnn":
            prediction = rnn_model.predict(input_data)
        else:
            return jsonify({"error": "Invalid model type. Use 'lstm' or 'rnn'."}), 400

        prediction_scaled = prediction.reshape(-1,1)
        original_features = np.zeros((1,len(numerical_cols)))
        original_features[0,1] = prediction_scaled[0,0] #Insert the prediction into the correct feature index.
        inversed_prediction = scaler.inverse_transform(original_features)
        return float(inversed_prediction[0,1])

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    date = data.get("date")
    city = data.get("city")
    model_type = data.get("model", "lstm")

    if not date or not city:
        return jsonify({"error": "Please provide date and city"}), 400

    prediction = predict_temperature(date, city, model_type)

    if isinstance(prediction, tuple): #handles errors returned by predict_temperature.
      return prediction

    return jsonify({"city": city, "date": date, "predicted_temperature": prediction, "model": model_type})

if __name__ == '__main__':
    app.run(debug=True)