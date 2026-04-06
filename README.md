# 🌤️ Polymarket Weather Trading Agent

An autonomous AI agent that predicts weather outcomes and executes trades on Polymarket with real-time decision-making and continuous learning.

## 📋 Project Overview

This system:
- Collects real-time weather data from multiple sources
- Uses machine learning (XGBoost) to predict weather events
- Maps predictions to Polymarket markets
- Executes profitable trades based on probability edge
- Manages risk and learns from historical data
- Provides a real-time dashboard

## 🏗️ Project Structure

```
polymarket-weather-agent/
├── config/
│   ├── config.yaml              # Main configuration
│   └── markets.json             # Polymarket mapping
├── src/
│   ├── data_collection/
│   │   ├── __init__.py
│   │   ├── weather_api.py       # Multi-source weather data
│   │   └── data_fetcher.py      # Main data collection module
│   ├── feature_engineering/
│   │   ├── __init__.py
│   │   └── features.py          # Feature generation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── training.py          # Model training pipeline
│   │   └── predictor.py         # Inference module
│   ├── market_mapping/
│   │   ├── __init__.py
│   │   └── mapper.py            # Map predictions to markets
│   ├── trading/
│   │   ├── __init__.py
│   │   ├── strategy.py          # Trading logic
│   │   ├── executor.py          # Order execution
│   │   └── risk_manager.py      # Risk controls
│   ├── learning/
│   │   ├── __init__.py
│   │   └── feedback_loop.py     # Model retraining
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py            # Logging utilities
│   │   └── helpers.py           # Helper functions
│   └── main.py                  # Entry point
├── dashboard/
│   ├── app.py                   # Flask app
│   ├── templates/
│   │   ├── index.html
│   │   └── trade_history.html
│   └── static/
│       └── style.css
├── data/
│   ├── raw/                     # Raw weather data
│   ├── processed/               # Processed features
│   ├── models/                  # Trained models
│   └── history/                 # Trading history
├── tests/
│   ├── test_models.py
│   ├── test_strategy.py
│   └── test_data.py
├── requirements.txt             # Dependencies
├── .env.example                 # Environment variables template
└── setup.py                     # Installation script
```

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
```

### 2. Configure
Edit `config/config.yaml` with:
- Weather API keys
- Polymarket settings
- Trading parameters

### 3. Run Agent
```bash
python src/main.py
```

### 4. View Dashboard
```bash
python dashboard/app.py
# Visit http://localhost:5000
```

## 📚 Core Modules

### Data Collection
Fetches weather data from OpenWeatherMap, WeatherAPI, and optional NOAA feeds.

### Feature Engineering
Generates 50+ features including:
- Moving averages (3h, 6h, 12h, 24h)
- Rain probability trends
- Temperature anomalies
- Seasonality indicators

### Machine Learning
XGBoost model that predicts:
- Rain probability (0-100%)
- Temperature thresholds
- Extreme weather events

### Market Mapping
Automatically maps predictions to Polymarket questions with:
- Location encoding
- Time resolution matching
- Confidence scoring

### Trading Strategy
Executes trades when:
- Edge > threshold (default 5%)
- Position limits respected
- Risk limits satisfied

### Risk Management
- Daily loss limits
- Position concentration limits
- Emergency shutdown protocols

## ⚙️ Configuration

See `config/config.yaml` for all settings:
```yaml
weather:
  api_keys:
    openweathermap: your_key
    weatherapi: your_key
  refresh_interval: 3600  # seconds

trading:
  edge_threshold: 0.05    # 5% edge
  max_risk_per_trade: 0.02  # 2% of capital
  kelly_fraction: 0.25
  
polymarket:
  api_url: https://api.polymarket.com
  slippage_tolerance: 0.02
```

## 📊 Performance Metrics

Track via dashboard:
- Prediction accuracy
- Win rate
- Sharpe ratio
- Max drawdown
- Total PnL

## 🔐 Security

- API keys stored in .env
- No private keys in code
- Input validation on all APIs
- Rate limiting

## 📖 Documentation

- [Data Collection Guide](docs/data_collection.md)
- [Model Training](docs/model_training.md)
- [Trading Strategy](docs/trading_strategy.md)
- [Deployment](docs/deployment.md)

## 🛠️ Development

Run tests:
```bash
pytest tests/
```

## 📄 License

MIT
