import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [ticker, setTicker] = useState('AAPL');
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    setLoading(true);
    try {
      const res = await axios.post('http://127.0.0.1:5000/predict', { ticker });
      setPrediction(res.data);
    } catch (err) {
      console.error(err);
      alert('Error fetching prediction');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>Stock Price Predictor (LSTM)</h1>
      <input
        type="text"
        value={ticker}
        onChange={(e) => setTicker(e.target.value)}
        placeholder="Enter stock symbol (e.g., AAPL)"
      />
      <button onClick={handlePredict} disabled={loading}>
        {loading ? 'Predicting...' : 'Predict'}
      </button>
  
      {prediction && (
        <div className="result">
          <h2>📈 Prediction for {ticker.toUpperCase()}:</h2>
          <p>Next predicted price: <strong>${prediction.price.toFixed(2)}</strong></p>
          <img
            src={`http://127.0.0.1:5000/plot.png?${Date.now()}`}
            alt="Prediction plot"
            style={{ width: '60%', border: '1px solid #ccc', borderRadius: '8px', marginTop: '20px' }}
          />
        </div>
      )}
  
      {/* 🔍 How it works section */}
      <div className="info" style={{ marginTop: '50px', textAlign: 'left', maxWidth: '700px', marginInline: 'auto' }}>
        <h2>🧠 How It Works</h2>
        <p>
          This tool uses a <strong>Long Short-Term Memory (LSTM)</strong> neural network to predict the next price of a
          stock based on recent trends.
        </p>
        <p>
          The model looks at the last 60 hourly closing prices, and also incorporates recent
          <strong> sentiment data</strong> scraped from Yahoo Finance headlines for the stock.
        </p>
        <p>
          Sentiment is quantified using <strong>TextBlob</strong> — a Natural Language Processing tool — and its average
          polarity is combined with price data to inform the LSTM. This way, the model considers both numerical and emotional
          market signals when making a prediction.
        </p>
        <p>
          The result is a predicted next price and a chart that visualizes the recent price movement with the model's
          predicted value overlaid.
        </p>
      </div>
    </div>
  );
  
}

export default App;
