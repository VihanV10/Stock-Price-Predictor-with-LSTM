# 📈 Stock Price Predictor with LSTM + Sentiment (Flask + React)

This project is a full-stack web application that predicts the next-hour price of any publicly traded stock by combining historical stock data with real-time news sentiment. It uses a Long Short-Term Memory (LSTM) neural network and is powered by a Python Flask backend and a React frontend.

---

## 🧠 Project Overview

### 🎯 Goal
To build an AI-powered system that:
- Fetches live stock data using `yfinance`
- Analyzes recent news sentiment using NLP
- Feeds both price and sentiment into an LSTM model
- Returns the predicted next price with a visualization

---

## 🔧 Tech Stack

| Layer         | Technology                            |
|---------------|----------------------------------------|
| Frontend      | React (Hooks, Axios)                  |
| Backend       | Flask + Flask-CORS                    |
| ML Model      | TensorFlow/Keras (LSTM)               |
| Sentiment     | TextBlob + BeautifulSoup (Web Scraping) |
| Data Source   | Yahoo Finance (via `yfinance`)        |
| Visualization | Matplotlib                            |

---

## 🧩 Architecture

### 🔁 Workflow:

1. **User Inputs Stock Ticker** (e.g., AAPL) via React UI.
2. **React sends a POST request** to the Flask API (`/predict`).
3. **Flask Backend**:
   - Fetches 60 days of hourly closing prices using `yfinance`
   - Scrapes recent Yahoo Finance headlines for the ticker
   - Analyzes the headlines using `TextBlob` to extract a sentiment score
   - Prepares an LSTM model using both price & sentiment
   - Trains the model and predicts the next price
   - Generates and saves a plot with the prediction
4. **React receives prediction data** and renders:
   - The predicted next price
   - A plot of recent prices + predicted line
   - A section explaining the model logic

---

## 📊 Model Details

The backend uses a **2-feature LSTM**:
- **Feature 1**: Scaled historical closing prices (lookback = 60 timesteps)
- **Feature 2**: Repeated average sentiment polarity from recent financial headlines

> Example input shape: `(60, 2)` — 60 time steps with `[price, sentiment]` at each step

This hybrid feature input allows the model to learn not just numerical trends, but also public sentiment momentum.

---

Sample Run:<img width="1470" alt="Screenshot 2025-05-17 at 8 15 58 PM" src="https://github.com/user-attachments/assets/2d0627dd-bd5e-4e5d-b070-bd7274229795" />
