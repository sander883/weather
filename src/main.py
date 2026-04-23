#!/usr/bin/env python3
"""Main entry point for Polymarket Weather Trading Agent."""

import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, Any
import os
from pathlib import Path
import sys

# Load environment variables from .env file FIRST, before any other imports
# This ensures all dependencies see the environment variables
project_root = Path(__file__).parent.parent
env_file = project_root / '.env'

# Load .env manually to ensure it works
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()
    sys.stdout.write(f"✓ Manually loaded .env from: {env_file}\n")
else:
    sys.stdout.write(f"✗ .env file not found at: {env_file}\n")

# Verify API keys are loaded
owm_key = os.getenv('OPENWEATHERMAP_API_KEY')
weather_key = os.getenv('WEATHERAPI_KEY')
sys.stdout.write(f"OPENWEATHERMAP_API_KEY: {owm_key[:10] if owm_key else 'NOT FOUND'}...\n")
sys.stdout.write(f"WEATHERAPI_KEY: {weather_key[:10] if weather_key else 'NOT FOUND'}...\n")
sys.stdout.flush()

from src.utils.logger import get_logger
from src.utils.helpers import load_config, load_markets
from src.data_collection.data_fetcher import DataFetcher
from src.feature_engineering.features import FeatureEngineer
from src.models.training import ModelTrainer
from src.models.predictor import Predictor
from src.market_mapping.mapper import MarketMapper
from src.trading.strategy import TradingStrategy
from src.trading.executor import OrderExecutor
from src.trading.risk_manager import RiskManager
from src.learning.feedback_loop import LearningLoop

logger = get_logger(__name__)


class PolymarketWeatherAgent:
    """Main autonomous trading agent."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the agent."""
        if config is None:
            config = load_config()

        self.config = config
        self.data_fetcher = DataFetcher(config)
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer(config)
        self.predictor = Predictor(self.model_trainer)
        self.market_mapper = MarketMapper()
        self.trading_strategy = TradingStrategy(config)
        self.executor = OrderExecutor(config, simulate=True)  # Start with simulation
        self.risk_manager = RiskManager(config)
        self.learning_loop = LearningLoop(config)

        self.running = False
        self.locations = [loc['name'] for loc in config.get('weather', {}).get('locations', [])]

        logger.info("Agent initialized successfully")

    def train_initial_model(self, location: str = 'New York') -> bool:
        """Train initial model for a location."""
        logger.info(f"Training initial model for {location}")

        try:
            # Populate historical data (7 days)
            logger.info(f"Populating historical data for {location}...")
            record_count = self.data_fetcher.populate_historical_data(location, days=7)

            if record_count == 0:
                logger.warning(f"No historical data could be fetched for {location}")
                logger.info("Will continue with baseline predictions until data accumulates")
                return False

            logger.info(f"Successfully populated {record_count} historical records for {location}")

            # Load historical data
            features, target = self.data_fetcher.prepare_training_data(location, days=30)

            if features.empty:
                logger.warning(f"No historical data yet for {location} - will retrain after data collection")
                return False

            # Engineer features
            features = self.feature_engineer.engineer_features(features)

            if features.empty:
                logger.error("Feature engineering produced empty DataFrame")
                return False

            # Split data
            X_train, X_test, y_train, y_test = self.model_trainer.prepare_training_data(
                features, target, test_size=0.2
            )

            # Train
            metrics = self.model_trainer.train(X_train, y_train, X_test, y_test)

            if metrics:
                logger.info(f"Model trained successfully: {metrics}")
                self.model_trainer.save_model()
                self.predictor.model_trainer = self.model_trainer
                return True

            return False

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return False

    def get_predictions(self) -> Dict[str, Any]:
        """Get predictions for all locations."""
        predictions = {}

        for location in self.locations:
            try:
                # Fetch current weather
                weather = self.data_fetcher.fetch_current_weather(location)

                if not weather:
                    logger.warning(f"Could not fetch weather for {location}")
                    continue

                # Get prediction
                prediction = self.predictor.predict(weather)

                if prediction:
                    predictions[location] = prediction
                    logger.info(f"Prediction for {location}: {prediction['rain_probability']:.2%}")

            except Exception as e:
                logger.error(f"Error getting prediction for {location}: {e}")

        return predictions

    def find_trading_opportunities(self, predictions: Dict[str, Any]) -> list:
        """Find trading opportunities based on predictions."""
        opportunities = []

        try:
            # Find opportunities across all markets
            all_opportunities = self.market_mapper.find_opportunities(
                predictions,
                edge_threshold=self.config.get('trading', {}).get('edge_threshold', 0.05)
            )

            # Defensive: ensure we have a list of dicts
            if not all_opportunities:
                logger.debug("No opportunities found")
                return []

            if not isinstance(all_opportunities, list):
                logger.error(f"Expected list of opportunities, got {type(all_opportunities)}")
                return []

            # Rank by attractiveness
            ranked = self.trading_strategy.rank_opportunities(all_opportunities)

            # Check feasibility with risk manager
            for opp in ranked:
                # Defensive: ensure opp is a dict
                if not isinstance(opp, dict):
                    logger.warning(f"Skipping non-dict opportunity: {type(opp)}")
                    continue

                trade = self.trading_strategy.evaluate_opportunity(opp)

                # Defensive: ensure trade is a dict before using it
                if not isinstance(trade, dict):
                    logger.warning(f"evaluate_opportunity returned non-dict: {type(trade)}")
                    continue

                if trade:
                    is_feasible, reason = self.risk_manager.check_trade_feasibility(
                        trade,
                        self.executor.get_open_positions()
                    )

                    if is_feasible:
                        opportunities.append(trade)
                    else:
                        logger.debug(f"Trade rejected: {reason}")

            logger.info(f"Found {len(opportunities)} trading opportunities")
            return opportunities

        except Exception as e:
            logger.error(f"Error finding opportunities: {e}", exc_info=True)
            return []

    def execute_trades(self, trades: list) -> None:
        """Execute trades."""
        if not trades:
            logger.debug("No trades to execute")
            return

        for trade in trades:
            try:
                # Defensive: ensure trade is a dict
                if not isinstance(trade, dict):
                    logger.error(f"Invalid trade type {type(trade)}, skipping")
                    continue

                # Validate required fields
                if 'market_id' not in trade or 'action' not in trade:
                    logger.error(f"Trade missing required fields: {list(trade.keys())}")
                    continue

                logger.info(f"Executing trade: {trade['market_id']} - {trade['action']}")

                # Execute
                execution = self.executor.execute_trade(trade)

                if execution.get('status') == 'FILLED':
                    logger.info(f"Trade executed: {execution['trade_id']}")
                    self.learning_loop.record_prediction(
                        {'rain_probability': trade.get('implied_probability')},
                        {'id': trade['market_id'], 'question': trade.get('question'),
                         'location': trade.get('location')}
                    )
                else:
                    logger.warning(f"Trade execution failed: {execution}")

            except Exception as e:
                logger.error(f"Error executing trade: {e}", exc_info=True)

    def update_positions(self) -> None:
        """Update open positions and check for exits."""
        open_positions = self.executor.get_open_positions()

        if not open_positions:
            return

        logger.info(f"Updating {len(open_positions)} open positions")

        for position in open_positions:
            # Check exit conditions
            exit_reason = self.trading_strategy.should_exit_position(position)

            if exit_reason:
                logger.info(f"Closing position {position['trade_id']}: {exit_reason}")

                # Use last known price as exit price
                exit_price = position.get('entry_price', 0.5)
                close_result = self.executor.close_position(position['trade_id'], exit_price)

                if close_result.get('status') == 'CLOSED':
                    self.risk_manager.record_close(close_result)
                    self.learning_loop.record_outcome(
                        position['market_id'],
                        close_result.get('pnl', 0) > 0
                    )

    def check_retraining(self) -> None:
        """Check if model should be retrained."""
        if self.learning_loop.should_retrain():
            logger.info("Triggering model retraining")
            success = self.learning_loop.retrain_model()

            if success:
                # Load new model
                latest_model = self.model_trainer.get_latest_model()
                if latest_model:
                    self.model_trainer.load_model(latest_model)
                    self.predictor.model_trainer = self.model_trainer

    def print_status(self) -> None:
        """Print agent status."""
        portfolio = self.risk_manager.get_portfolio_stats()
        execution = self.executor.get_execution_stats()
        learning = self.learning_loop.get_learning_status()

        logger.info("=" * 60)
        logger.info("AGENT STATUS")
        logger.info("=" * 60)
        logger.info(f"Time: {datetime.now(timezone.utc).isoformat()}")
        logger.info(f"Capital: ${portfolio.get('current_capital', 0):.2f} "
                   f"(Change: {portfolio.get('total_pnl_percent', 0):.2f}%)")
        logger.info(f"Trades: {execution.get('total_trades', 0)} "
                   f"(Win rate: {execution.get('win_rate', 0):.2%})")
        logger.info(f"Open positions: {execution.get('open_positions', 0)}")
        logger.info(f"Circuit breaker: {'TRIGGERED' if portfolio.get('circuit_breaker_triggered') else 'OK'}")
        logger.info(f"Last retrain: {learning.get('last_retrain', 'Never')}")
        logger.info("=" * 60)

    def run_once(self) -> None:
        """Run a single iteration of the trading loop."""
        try:
            logger.info("Starting trading cycle")

            # Get predictions
            try:
                predictions = self.get_predictions()
                if not predictions:
                    logger.warning("No predictions available")
                    return
            except Exception as e:
                logger.error(f"Error getting predictions: {e}", exc_info=True)
                return

            # Find opportunities
            try:
                trades = self.find_trading_opportunities(predictions)
                if not isinstance(trades, list):
                    logger.error(f"find_trading_opportunities returned {type(trades)}, expected list")
                    return
            except Exception as e:
                logger.error(f"Error finding opportunities: {e}", exc_info=True)
                return

            # Execute trades
            try:
                if trades:
                    if not all(isinstance(t, dict) for t in trades):
                        bad_trades = [t for t in trades if not isinstance(t, dict)]
                        logger.error(f"Invalid trades in list: {[type(t) for t in bad_trades]}")
                        trades = [t for t in trades if isinstance(t, dict)]

                    if trades:
                        self.execute_trades(trades)
            except Exception as e:
                logger.error(f"Error executing trades: {e}", exc_info=True)

            # Update positions
            try:
                self.update_positions()
            except Exception as e:
                logger.error(f"Error updating positions: {e}", exc_info=True)

            # Check retraining
            try:
                self.check_retraining()
            except Exception as e:
                logger.error(f"Error checking retraining: {e}", exc_info=True)

            # Print status
            try:
                self.print_status()
            except Exception as e:
                logger.error(f"Error printing status: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Unexpected error in trading cycle: {e}", exc_info=True)

    def run(self, interval: int = 3600) -> None:
        """
        Run the agent continuously.

        Args:
            interval: Seconds between iterations
        """
        self.running = True
        logger.info(f"Starting agent with {interval}s interval")

        # Try to train initial model (may skip on first run if no data)
        self.train_initial_model()
        logger.info("Model training attempt complete")

        try:
            iteration = 0
            while self.running:
                try:
                    iteration += 1
                    logger.info(f"--- Iteration {iteration} ---")
                    self.run_once()

                    # Check for emergency shutdown
                    if self.risk_manager.check_emergency_shutdown():
                        logger.critical("Emergency shutdown triggered")
                        break

                    # Wait before next iteration
                    logger.info(f"Sleeping for {interval} seconds")
                    time.sleep(interval)

                except KeyboardInterrupt:
                    logger.info("Keyboard interrupt received")
                    break
                except Exception as e:
                    logger.error(f"Error in iteration {iteration}: {e}")
                    time.sleep(interval)

        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Shutdown the agent gracefully."""
        logger.info("Shutting down agent")
        self.running = False

        # Export learning data
        self.learning_loop.export_learning_data()

        # Print final status
        self.print_status()

        logger.info("Agent shutdown complete")


if __name__ == '__main__':
    # Initialize and run agent
    agent = PolymarketWeatherAgent()

    # Run with 1-hour interval (3600 seconds)
    # For testing, use smaller interval like 60 seconds
    agent.run(interval=60)
