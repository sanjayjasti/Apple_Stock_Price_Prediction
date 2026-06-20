from flask import Flask, render_template
import yfinance as yf
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Load model
model = load_model("model.keras")

# Load scaler
scaler = pickle.load(open("scaler.pkl", "rb"))


@app.route("/")
def home():

    # Get stock data
    ticker = yf.Ticker("AAPL")
    df = ticker.history(period="2y")

    # Close prices
    data = df["Close"].values.reshape(-1, 1)

    # Scale
    scaled_data = scaler.transform(data)

    # Create sequences
    x_test = []
    y_test = []

    sequence_length = 60

    for i in range(sequence_length, len(scaled_data)):
        x_test.append(scaled_data[i-sequence_length:i])
        y_test.append(scaled_data[i])

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # Predict historical values
    predictions = model.predict(x_test, verbose=0)

    predictions = scaler.inverse_transform(predictions)

    y_test = scaler.inverse_transform(y_test)

    # Dates
    dates = df.index[sequence_length:]

    # Current stock price
    current_price = float(data[-1][0])

    # Tomorrow prediction
    last_sequence = scaled_data[-60:]
    last_sequence = last_sequence.reshape(1, 60, 1)

    future_scaled = model.predict(last_sequence, verbose=0)

    future_price = scaler.inverse_transform(future_scaled)

    predicted_tomorrow = float(future_price[0][0])

    # Direction
    if predicted_tomorrow > current_price:
        direction = "📈 Up"
    else:
        direction = "📉 Down"

    # Save graph
    plt.figure(figsize=(12, 6))

    plt.plot(dates, y_test, label="Actual Price")

    plt.plot(dates, predictions, label="Predicted Price")

    plt.title("Actual vs Predicted Stock Price")

    plt.xlabel("Date")

    plt.ylabel("Price")

    plt.legend()

    plt.grid(True)

    graph_path = os.path.join("static", "graph.png")

    plt.savefig(graph_path)

    plt.close()

    # Convert to list for HTML
    actual_prices = y_test.flatten().tolist()

    predicted_prices = predictions.flatten().tolist()

    dates = [str(date.date()) for date in dates]

    return render_template(
        "index.html",
        dates=dates,
        actual_prices=actual_prices,
        predicted_prices=predicted_prices,
        current_price=round(current_price, 2),
        predicted_tomorrow=round(predicted_tomorrow, 2),
        direction=direction
    )

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)