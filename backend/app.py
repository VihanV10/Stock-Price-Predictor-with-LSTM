from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from flask_cors import CORS
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    ticker = data.get("ticker", "AAPL")
    print(f"\n📥 Received prediction request for ticker: {ticker}")

    try:
        # --- STOCK PRICE DATA ---
        print("🔄 Downloading stock data...")
        df = yf.download(ticker, period="60d", interval="1h")[['Close']].tail(500).dropna()
        print(f"✅ Downloaded {len(df)} data points.")

        # --- SENTIMENT DATA ---
        print("📰 Scraping news for sentiment...")
        news_url = f"https://finance.yahoo.com/quote/{ticker}?p={ticker}"
        response = requests.get(news_url)
        soup = BeautifulSoup(response.text, "lxml")
        headlines = [h.get_text() for h in soup.find_all('h3')]
        sentiment_scores = [TextBlob(h).sentiment.polarity for h in headlines]
        avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
        print(f"🧠 Average sentiment: {avg_sentiment:.2f}")

        # --- PREPROCESSING ---
        print("⚙️ Scaling and preparing data...")
        scaler = MinMaxScaler()
        price_scaled = scaler.fit_transform(df)

        LOOKBACK = 60
        X, y = [], []

        for i in range(LOOKBACK, len(price_scaled)):
            price_window = price_scaled[i - LOOKBACK:i]  # shape: (60, 1)
            sentiment_window = np.full((LOOKBACK, 1), avg_sentiment)  # shape: (60, 1)
            combined = np.hstack((price_window, sentiment_window))  # shape: (60, 2)
            X.append(combined)
            y.append(price_scaled[i])

        X, y = np.array(X), np.array(y)
        print(f"📊 Created {X.shape[0]} training samples with shape {X.shape[1:]}")

        # --- MODEL ---
        print("🧠 Training LSTM model...")
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(LOOKBACK, 2)),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=3, batch_size=32, verbose=0)
        print("✅ Model trained.")

        # --- PREDICTION ---
        last_price_seq = price_scaled[-LOOKBACK:]
        last_sentiment_seq = np.full((LOOKBACK, 1), avg_sentiment)
        last_combined = np.hstack((last_price_seq, last_sentiment_seq)).reshape(1, LOOKBACK, 2)

        pred_scaled = model.predict(last_combined)
        prediction = scaler.inverse_transform(pred_scaled)[0][0]
        print(f"📈 Predicted next price: ${prediction:.2f}")

        # --- PLOT ---
        print("📉 Plotting...")
        plt.figure(figsize=(10, 4))
        plt.plot(df.index[-100:], df['Close'].values[-100:], label="Real Price")
        plt.axhline(prediction, color='red', linestyle='--', label='Predicted')
        plt.title(f"{ticker} LSTM Prediction (w/ Sentiment)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("plot.png")
        plt.close()
        print("✅ Plot saved")

        return jsonify({"price": float(prediction)})

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/plot.png", methods=["GET"])
def get_plot():
    print("📤 Sending plot.png to frontend...")
    return send_file("plot.png", mimetype="image/png")

if __name__ == "__main__":
    print("🚀 Starting Flask app...")
    app.run(debug=True)
