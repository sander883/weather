# Quick Start (5 Minutes)

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Set Up API Keys

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```
OPENWEATHERMAP_API_KEY=sk_live_xxxxx
WEATHERAPI_KEY=xxxxxxxxxxxx
```

Get free API keys:
- OpenWeatherMap: https://openweathermap.org/api
- WeatherAPI: https://www.weatherapi.com/

## Step 3: Run the Agent

```bash
python src/main.py
```

You should see:

```
2024-04-06 10:30:45 - src.main - INFO - Agent initialized successfully
2024-04-06 10:30:45 - src.main - INFO - Training initial model for New York
...
--- Iteration 1 ---
Prediction for New York: 45.23%
Found 2 trading opportunities
...
```

## Step 4: View Dashboard (Optional)

In another terminal:

```bash
python dashboard/app.py
```

Then open: http://localhost:5000

## What It Does

1. **Fetches weather data** from multiple sources
2. **Trains ML model** on historical weather data
3. **Makes predictions** about weather events
4. **Finds trading opportunities** on Polymarket
5. **Executes trades** (in simulation mode by default)
6. **Tracks performance** and learns continuously

## Example Output

```
============================================================
AGENT STATUS
============================================================
Time: 2024-04-06T10:35:22.123456
Capital: $10,000.00 (Change: 0.00%)
Trades: 3 (Win rate: 66.67%)
Open positions: 1
Circuit breaker: OK
Last retrain: Never
============================================================
```

## Configuration

Edit `config/config.yaml` to:
- Add more locations
- Adjust trading parameters
- Change model settings
- Modify risk limits

## Next Steps

1. **Monitor the dashboard** at http://localhost:5000
2. **Check logs** in `logs/agent.log`
3. **Review performance** in `data/history/`
4. **Adjust settings** in `config/config.yaml`
5. **Add more markets** in `config/markets.json`

## Real Trading

⚠️ **Agent runs in SIMULATION mode by default**

To enable real trading:

1. Get Polymarket API credentials
2. Set `POLYMARKET_PRIVATE_KEY` in `.env`
3. Change `simulate=True` to `simulate=False` in `src/main.py`
4. **Start small** - use min position sizes
5. **Monitor closely** - check dashboard every hour

## Troubleshooting

**"No module named 'src'"**
```bash
cd polymarket-weather-agent  # Make sure you're in project root
python src/main.py
```

**"API key not found"**
```bash
# Check .env file exists
cat .env
# Make sure keys are set correctly
```

**"No weather data available"**
```bash
# Check internet connection
# Verify API keys are valid
# Check logs for specific errors
tail -f logs/agent.log
```

## Success Indicators

✅ Agent starts and trains model
✅ Weather predictions are generated
✅ Trading opportunities are found
✅ Dashboard loads at http://localhost:5000
✅ Trades are executed (in simulation)
✅ Performance is tracked

## Performance Targets

- **Accuracy**: 55%+ (better than random)
- **Win Rate**: 50%+ (frequency of profitable trades)
- **Sharpe Ratio**: 1.0+ (risk-adjusted returns)
- **Max Drawdown**: <20% (largest portfolio decline)

## Support

- Check logs: `tail -f logs/agent.log`
- Read docs: See SETUP_GUIDE.md
- Review code: Check src/ directory
- Ask questions: See README.md for contact info

Good luck! 🚀
