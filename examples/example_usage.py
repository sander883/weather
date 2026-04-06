#!/usr/bin/env python3
"""
Example usage of the Polymarket Weather Trading Agent.

This script demonstrates how to use the agent components programmatically.
"""

import sys
sys.path.insert(0, '..')

from src.utils.helpers import load_config
from src.data_collection import DataFetcher
from src.feature_engineering import FeatureEngineer
from src.models import ModelTrainer, Predictor
from src.market_mapping import MarketMapper
from src.trading import TradingStrategy, OrderExecutor, RiskManager
from src.learning import LearningLoop


def example_1_fetch_weather():
    """Example 1: Fetch weather data."""
    print("\n" + "="*60)
    print("Example 1: Fetching Weather Data")
    print("="*60)

    config = load_config()
    fetcher = DataFetcher(config)

    # Fetch for New York
    weather = fetcher.fetch_current_weather('New York')
    if weather:
        print(f"Location: {weather['location']}")
        print(f"Temperature: {weather['temperature']:.1f}°C")
        print(f"Humidity: {weather['humidity']:.0f}%")
        print(f"Wind Speed: {weather['wind_speed']:.1f} m/s")
        print(f"Clouds: {weather['clouds']:.0f}%")

    # Get forecast
    forecast = fetcher.fetch_forecast('New York', days=1)
    if forecast:
        print(f"\nForecast: {len(forecast)} time steps")
        for item in forecast[:3]:
            print(f"  {item['timestamp']}: Temp {item['temperature']:.1f}°C, "
                  f"Rain {item['rain_probability']*100:.0f}%")


def example_2_feature_engineering():
    """Example 2: Generate features."""
    print("\n" + "="*60)
    print("Example 2: Feature Engineering")
    print("="*60)

    config = load_config()
    fetcher = DataFetcher(config)
    engineer = FeatureEngineer()

    # Load historical data
    print("Loading historical data...")
    features, target = fetcher.prepare_training_data('New York', days=7)

    if features.empty:
        print("No historical data available")
        return

    print(f"Raw features shape: {features.shape}")

    # Engineer features
    print("Engineering features...")
    engineered = engineer.engineer_features(features)

    print(f"Engineered features shape: {engineered.shape}")
    print(f"Feature columns: {engineered.columns.tolist()[:10]}... (showing first 10)")


def example_3_train_model():
    """Example 3: Train a model."""
    print("\n" + "="*60)
    print("Example 3: Training Model")
    print("="*60)

    config = load_config()
    fetcher = DataFetcher(config)
    trainer = ModelTrainer(config)

    # Load data
    print("Loading training data...")
    features, target = fetcher.prepare_training_data('New York', days=30)

    if features.empty:
        print("No training data available - skipping training")
        return

    # Split and train
    print("Training model...")
    X_train, X_test, y_train, y_test = trainer.prepare_training_data(features, target)
    metrics = trainer.train(X_train, y_train, X_test, y_test)

    if metrics:
        print(f"Training complete!")
        print(f"  Train Accuracy: {metrics.get('train_accuracy', 0):.3f}")
        print(f"  Validation Accuracy: {metrics.get('val_accuracy', 0):.3f}")
        print(f"  Train AUC: {metrics.get('train_auc', 0):.3f}")
        print(f"  Validation AUC: {metrics.get('val_auc', 0):.3f}")

        # Save model
        model_path = trainer.save_model('example_model')
        print(f"  Model saved to: {model_path}")


def example_4_make_predictions():
    """Example 4: Make predictions."""
    print("\n" + "="*60)
    print("Example 4: Making Predictions")
    print("="*60)

    config = load_config()
    fetcher = DataFetcher(config)
    trainer = ModelTrainer(config)
    predictor = Predictor(trainer)

    # Load and train model first
    print("Training model...")
    features, target = fetcher.prepare_training_data('New York', days=7)
    if not features.empty:
        X_train, X_test, y_train, y_test = trainer.prepare_training_data(features, target)
        trainer.train(X_train, y_train, X_test, y_test)

        # Get current weather
        print("Fetching current weather...")
        weather = fetcher.fetch_current_weather('New York')

        if weather:
            # Make prediction
            print("Making prediction...")
            prediction = predictor.predict(weather)

            if prediction:
                print(f"Prediction for {weather['location']}:")
                print(f"  Rain Probability: {prediction['rain_probability']*100:.1f}%")
                print(f"  No Rain Probability: {prediction['no_rain_probability']*100:.1f}%")
                print(f"  Prediction: {prediction['prediction'].upper()}")
                print(f"  Confidence: {prediction['confidence']*100:.1f}%")


def example_5_market_mapping():
    """Example 5: Map predictions to markets."""
    print("\n" + "="*60)
    print("Example 5: Market Mapping")
    print("="*60)

    mapper = MarketMapper()

    # Get market summary
    print("Available markets:")
    summary = mapper.get_market_summary()
    print(summary)

    # Validate markets
    print("\nValidating market configuration...")
    validation = mapper.validate_market_mapping()
    print(f"Total markets: {validation['total_markets']}")
    print(f"Enabled markets: {validation['enabled_markets']}")
    print(f"Markets by location: {validation['by_location']}")

    if validation['issues']:
        print("Issues found:")
        for issue in validation['issues']:
            print(f"  - {issue}")


def example_6_trading_strategy():
    """Example 6: Evaluate trading opportunities."""
    print("\n" + "="*60)
    print("Example 6: Trading Strategy")
    print("="*60)

    config = load_config()
    strategy = TradingStrategy(config)

    # Create example opportunity
    opportunity = {
        'market_id': '0x123',
        'question': 'Will it rain in New York tomorrow?',
        'location': 'New York',
        'predicted_yes_probability': 0.72,
        'predicted_no_probability': 0.28,
        'current_yes_price': 0.55,
        'current_no_price': 0.45,
        'edge': 0.17,  # 17% edge
        'liquidity_usd': 5000,
        'volume_24h_usd': 1000
    }

    # Evaluate
    trade = strategy.evaluate_opportunity(opportunity)

    if trade:
        print(f"Trade recommendation:")
        print(f"  Market: {trade['question']}")
        print(f"  Action: {trade['action']}")
        print(f"  Position Size: ${trade['position_size']:.2f}")
        print(f"  Entry Price: {trade['entry_price']:.4f}")
        print(f"  Implied Probability: {trade['implied_probability']:.2%}")
        print(f"  Edge: {trade['edge']:.2%}")


def example_7_risk_management():
    """Example 7: Risk management."""
    print("\n" + "="*60)
    print("Example 7: Risk Management")
    print("="*60)

    config = load_config()
    risk_mgr = RiskManager(config)

    # Check portfolio stats
    stats = risk_mgr.get_portfolio_stats()
    print("Portfolio Statistics:")
    print(f"  Initial Capital: ${stats['initial_capital']:.2f}")
    print(f"  Current Capital: ${stats['current_capital']:.2f}")
    print(f"  Total P&L: ${stats['total_pnl']:.2f}")
    print(f"  Return: {stats['total_pnl_percent']:.2f}%")
    print(f"  Circuit Breaker: {'TRIGGERED' if stats['circuit_breaker_triggered'] else 'OK'}")


def example_8_learning_loop():
    """Example 8: Learning system."""
    print("\n" + "="*60)
    print("Example 8: Learning System")
    print("="*60)

    config = load_config()
    learning = LearningLoop(config)

    # Get learning status
    status = learning.get_learning_status()
    print("Learning System Status:")
    print(f"  Enabled: {status['enabled']}")
    print(f"  Should Retrain: {status['should_retrain']}")
    print(f"  Last Retrain: {status['last_retrain']}")
    print(f"  Prediction Records: {status['prediction_records']}")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Polymarket Weather Trading Agent - Examples")
    print("="*60)

    try:
        example_1_fetch_weather()
        example_2_feature_engineering()
        example_3_train_model()
        example_4_make_predictions()
        example_5_market_mapping()
        example_6_trading_strategy()
        example_7_risk_management()
        example_8_learning_loop()

        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
