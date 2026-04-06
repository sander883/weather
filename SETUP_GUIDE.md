# Polymarket Weather Trading Agent - Setup Guide

## 📋 Prerequisites

- Python 3.8+
- pip or conda
- PostgreSQL (optional, for persistent storage)
- Redis (optional, for caching)

## 🚀 Quick Start (5 minutes)

### 1. Clone and Install

```bash
cd polymarket-weather-agent
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys:
# - Get OpenWeatherMap key from https://openweathermap.org/api
# - Get WeatherAPI key from https://www.weatherapi.com
```

### 3. Initialize Configuration

The default `config/config.yaml` is already set up with:
- 3 locations (New York, Los Angeles, London)
- 5 example Polymarket markets
- XGBoost model settings
- Risk management defaults

### 4. Run the Agent

```bash
python src/main.py
```

Agent will:
1. Train initial model on historical data
2. Fetch weather data every 60 seconds
3. Generate predictions
4. Find trading opportunities
5. Execute trades (simulation mode by default)
6. Track performance

### 5. View Dashboard

In another terminal:

```bash
python dashboard/app.py
```

Open browser to: http://localhost:5000

## 📊 Detailed Setup

### A. API Key Configuration

#### OpenWeatherMap

1. Go to https://openweathermap.org/api
2. Sign up for free tier (5-day forecast)
3. Copy API key to `.env`:
   ```
   OPENWEATHERMAP_API_KEY=your_key_here
   ```

#### WeatherAPI

1. Go to https://www.weatherapi.com/
2. Sign up (free tier includes 1M calls/month)
3. Copy API key to `.env`:
   ```
   WEATHERAPI_KEY=your_key_here
   ```

#### Polymarket (Optional for Real Trading)

For actual trading on Polymarket:

1. Create Polymarket account
2. Generate API key from settings
3. For smart contract interaction:
   ```
   POLYMARKET_PRIVATE_KEY=your_private_key
   POLYMARKET_WALLET_ADDRESS=your_wallet
   ```

⚠️ **Never commit private keys to git!**

### B. Database Setup (Optional)

For persistent storage of trading history:

```bash
# Install PostgreSQL
# macOS: brew install postgresql
# Ubuntu: sudo apt-get install postgresql
# Windows: Download from https://www.postgresql.org/download/windows/

# Create database
createdb polymarket_weather

# Set connection in .env
DATABASE_URL=postgresql://user:password@localhost:5432/polymarket_weather
```

### C. Redis Setup (Optional)

For caching and task queuing:

```bash
# Install Redis
# macOS: brew install redis
# Ubuntu: sudo apt-get install redis-server
# Windows: Use WSL or Docker

# Start Redis
redis-server

# Set connection in .env
REDIS_URL=redis://localhost:6379/0
```

## 🔧 Configuration Tuning

### Adjust Trading Parameters

Edit `config/config.yaml`:

```yaml
trading:
  edge_threshold: 0.05        # Require 5% edge (higher = pickier)
  max_position_size: 1000     # Max per trade
  max_concurrent_positions: 5  # Max open trades

risk_management:
  portfolio:
    initial_capital: 10000
    max_daily_loss_percent: 2.0  # Stop after 2% daily loss
    max_drawdown_percent: 10.0   # Stop after 10% total loss
```

### Adjust Model Parameters

```yaml
model:
  xgboost_params:
    max_depth: 6               # Tree depth (higher = more complex)
    learning_rate: 0.05        # Slower learning = more stable
    n_estimators: 300          # More trees = more robust
```

### Add More Locations

Edit `config/config.yaml` `weather.locations`:

```yaml
weather:
  locations:
    - name: "Paris"
      lat: 48.8566
      lon: 2.3522
      country: FR
      timezone: Europe/Paris
```

Then add corresponding markets in `config/markets.json`.

## 📈 Running in Production

### Option 1: Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

ENV PYTHONUNBUFFERED=1
CMD ["python", "src/main.py"]
```

Build and run:

```bash
docker build -t polymarket-agent .
docker run -e OPENWEATHERMAP_API_KEY=xxx polymarket-agent
```

### Option 2: Linux/VPS Deployment

```bash
# SSH into VPS
ssh user@your_vps.com

# Install dependencies
sudo apt-get update
sudo apt-get install python3-pip python3-venv

# Clone repo
git clone https://github.com/user/polymarket-weather-agent
cd polymarket-weather-agent

# Setup virtualenv
python3 -m venv venv
source venv/bin/activate

# Install
pip install -r requirements.txt

# Create .env file
nano .env
# Paste your configuration

# Run with nohup (persists after disconnect)
nohup python src/main.py > agent.log 2>&1 &

# Check logs
tail -f agent.log
```

### Option 3: Systemd Service

Create `/etc/systemd/system/polymarket-agent.service`:

```ini
[Unit]
Description=Polymarket Weather Trading Agent
After=network.target

[Service]
Type=simple
User=trader
WorkingDirectory=/home/trader/polymarket-weather-agent
Environment="PATH=/home/trader/polymarket-weather-agent/venv/bin"
ExecStart=/home/trader/polymarket-weather-agent/venv/bin/python src/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable polymarket-agent
sudo systemctl start polymarket-agent
sudo systemctl status polymarket-agent
```

## 🧪 Testing

### Run Tests

```bash
pytest tests/ -v
```

### Backtest Strategy

Enable backtesting in `config/config.yaml`:

```yaml
backtest:
  enabled: true
  start_date: "2024-01-01"
  end_date: "2024-12-31"
  initial_capital: 10000
```

Then run:

```bash
python src/backtest.py
```

### Dry Run / Simulation

By default, the agent runs in simulation mode (no real trades):

```python
executor = OrderExecutor(config, simulate=True)  # Trades not real
```

Change to False for real trading (⚠️ use carefully!):

```python
executor = OrderExecutor(config, simulate=False)  # REAL TRADES
```

## 📊 Monitoring

### Dashboard

```bash
python dashboard/app.py
# Visit http://localhost:5000
```

### Logs

```bash
# View real-time logs
tail -f logs/agent.log

# View all today's logs
cat logs/agent.log | grep "$(date +%Y-%m-%d)"
```

### Metrics Export

```bash
# Export performance metrics
curl http://localhost:5000/api/status | jq
curl http://localhost:5000/api/performance | jq
curl http://localhost:5000/api/trades?limit=100 | jq
```

## 🔄 Continuous Learning

The agent automatically:

1. **Collects feedback** on all predictions
2. **Records outcomes** as markets resolve
3. **Calculates accuracy** metrics
4. **Retrains model** when:
   - 7 days have passed (default)
   - Accuracy drops >5%
   - 500+ new samples accumulated

Configure in `config/config.yaml`:

```yaml
learning:
  enabled: true
  retraining:
    trigger_conditions:
      model_age: 604800      # 7 days
      accuracy_drop: 0.05    # 5%
      new_samples: 500
```

## ⚠️ Risk Management

The agent has multiple safety mechanisms:

1. **Daily Loss Limit** - Stop trading if daily loss > 2%
2. **Max Drawdown** - Stop if portfolio drops > 10%
3. **Position Limits** - Max 5 open trades, max 2% per trade
4. **Circuit Breaker** - Emergency shutdown if conditions met
5. **Risk per Trade** - Kelly Criterion sizing

Configure in `config/config.yaml`:

```yaml
risk_management:
  circuit_breaker:
    daily_loss_threshold: 0.05        # 5% daily
    max_consecutive_losses: 5
    max_leverage: 1.0
```

## 🐛 Troubleshooting

### Issue: "No module named 'src'"

**Solution:**
```bash
# Make sure you're in the project root
cd polymarket-weather-agent
python src/main.py
```

### Issue: "API rate limit exceeded"

**Solution:**
```yaml
# Increase retry backoff in config.yaml
api:
  max_retries: 5
  retry_backoff: 2.0
```

### Issue: "Model training takes too long"

**Solution:**
```yaml
model:
  xgboost_params:
    n_estimators: 100  # Reduce from 300
    max_depth: 4       # Reduce from 6
```

### Issue: "No weather data available"

**Solution:**
- Check `.env` file has valid API keys
- Verify API keys have quota remaining
- Check network connectivity
- Look at logs for specific errors

### Issue: "Dashboard not loading"

**Solution:**
```bash
# Check Flask is running
ps aux | grep flask

# Try different port
python dashboard/app.py --port 8000
# Visit http://localhost:8000
```

## 📚 Documentation

- [Data Collection](docs/data_collection.md)
- [Feature Engineering](docs/feature_engineering.md)
- [Model Training](docs/model_training.md)
- [Trading Strategy](docs/trading_strategy.md)
- [Risk Management](docs/risk_management.md)
- [API Reference](docs/api_reference.md)

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repo
2. Create feature branch: `git checkout -b feature/amazing`
3. Make changes
4. Run tests: `pytest`
5. Submit pull request

## 📝 License

MIT License - see LICENSE file

## 🆘 Support

- Issues: https://github.com/user/polymarket-weather-agent/issues
- Discussions: https://github.com/user/polymarket-weather-agent/discussions
- Email: support@example.com

## 💡 Tips for Success

1. **Start with simulation** - Run in simulation mode for 1 week before real trades
2. **Monitor dashboard** - Check performance daily
3. **Review logs** - Look for warnings or errors
4. **Adjust edge threshold** - Start conservative (5-10%)
5. **Test on dry run** - Use small positions first
6. **Rebalance regularly** - Take profits periodically
7. **Keep learning** - Monitor model accuracy and retrain when needed

Good luck! 🚀
