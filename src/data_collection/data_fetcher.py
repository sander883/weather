"""Main data collection module."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
from pathlib import Path

from src.data_collection.weather_api import WeatherDataAggregator
from src.utils.logger import get_logger
from src.utils.helpers import load_config

logger = get_logger(__name__)


class DataFetcher:
    """Collect and manage weather data for trading."""

    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            config = load_config()

        self.config = config
        self.aggregator = WeatherDataAggregator(config)
        self.raw_data_dir = Path('data/raw')
        self.processed_data_dir = Path('data/processed')

        # Create directories
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

        # Store dataframes in memory
        self.current_data = {}
        self.historical_data = {}

    def fetch_current_weather(self, location: str) -> Optional[Dict]:
        """
        Fetch current weather for a location.

        Args:
            location: Location name from config

        Returns:
            Weather data dict or None
        """
        locations = {
            loc['name']: (loc['lat'], loc['lon'])
            for loc in self.config.get('weather', {}).get('locations', [])
        }

        if location not in locations:
            logger.error(f"Location {location} not found in config")
            return None

        lat, lon = locations[location]

        try:
            data = self.aggregator.get_weather_consensus(lat, lon)
            if data:
                self.current_data[location] = data
                self._save_raw_data(location, data)
                return data
        except Exception as e:
            logger.error(f"Error fetching weather for {location}: {e}")

        return None

    def fetch_forecast(self, location: str, days: int = 5) -> Optional[List[Dict]]:
        """
        Fetch weather forecast for a location.

        Args:
            location: Location name from config
            days: Number of days to forecast

        Returns:
            List of forecast dicts or None
        """
        locations = {
            loc['name']: (loc['lat'], loc['lon'])
            for loc in self.config.get('weather', {}).get('locations', [])
        }

        if location not in locations:
            logger.error(f"Location {location} not found in config")
            return None

        lat, lon = locations[location]

        try:
            data = self.aggregator.get_forecast_consensus(lat, lon, days)
            if data:
                return data
        except Exception as e:
            logger.error(f"Error fetching forecast for {location}: {e}")

        return None

    def fetch_all_locations(self) -> Dict[str, Dict]:
        """Fetch weather for all configured locations."""
        locations = self.config.get('weather', {}).get('locations', [])
        all_data = {}

        for location in locations:
            data = self.fetch_current_weather(location['name'])
            if data:
                all_data[location['name']] = data

        return all_data

    def populate_historical_data(self, location: str, days: int = 7) -> int:
        """
        Fetch and populate historical weather data for a location.

        Args:
            location: Location name from config
            days: Number of days of historical data to fetch

        Returns:
            Number of records fetched
        """
        locations = {
            loc['name']: (loc['lat'], loc['lon'])
            for loc in self.config.get('weather', {}).get('locations', [])
        }

        if location not in locations:
            logger.error(f"Location {location} not found in config")
            return 0

        lat, lon = locations[location]

        try:
            logger.info(f"Fetching {days} days of historical data for {location}...")
            historical_data = self.aggregator.get_historical_consensus(lat, lon, days)

            if not historical_data:
                logger.warning(f"No historical data available for {location}")
                return 0

            # Save all historical data points
            count = 0
            for data_point in historical_data:
                self._save_raw_data(location, data_point)
                count += 1

            logger.info(f"Populated {count} historical records for {location}")
            return count

        except Exception as e:
            logger.error(f"Error populating historical data for {location}: {e}")
            return 0

    def populate_all_historical_data(self, days: int = 7) -> Dict[str, int]:
        """
        Fetch historical data for all locations.

        Args:
            days: Number of days to fetch

        Returns:
            Dict mapping locations to number of records fetched
        """
        locations = self.config.get('weather', {}).get('locations', [])
        results = {}

        for location in locations:
            count = self.populate_historical_data(location['name'], days)
            results[location['name']] = count

        return results

    def _save_raw_data(self, location: str, data: Dict) -> None:
        """Save raw weather data to file."""
        try:
            timestamp = datetime.now().isoformat()
            filename = self.raw_data_dir / f"{location.lower().replace(' ', '_')}_{timestamp.split('T')[0]}.jsonl"

            with open(filename, 'a') as f:
                data['saved_at'] = timestamp
                f.write(json.dumps(data, default=str) + '\n')
        except Exception as e:
            logger.error(f"Error saving raw data: {e}")

    def load_historical_data(self, location: str, days: int = 30) -> pd.DataFrame:
        """
        Load historical weather data for a location.

        Args:
            location: Location name
            days: Number of days of history to load

        Returns:
            DataFrame with historical data
        """
        try:
            pattern = f"{location.lower().replace(' ', '_')}_*.jsonl"
            files = list(self.raw_data_dir.glob(pattern))

            if not files:
                logger.warning(f"No historical data found for {location}")
                return pd.DataFrame()

            # Load latest files
            all_records = []
            cutoff_date = datetime.now() - timedelta(days=days)

            for file in sorted(files)[-days:]:  # Get last N days
                try:
                    with open(file, 'r') as f:
                        for line in f:
                            record = json.loads(line)
                            if datetime.fromisoformat(record.get('timestamp', '')) > cutoff_date:
                                all_records.append(record)
                except Exception as e:
                    logger.warning(f"Error reading {file}: {e}")

            if not all_records:
                return pd.DataFrame()

            df = pd.DataFrame(all_records)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp'])

            logger.info(f"Loaded {len(df)} records for {location}")
            return df

        except Exception as e:
            logger.error(f"Error loading historical data for {location}: {e}")
            return pd.DataFrame()

    def prepare_training_data(self, location: str, days: int = 30) -> tuple:
        """
        Prepare data for model training.

        Returns:
            Tuple of (features_df, target_df)
        """
        df = self.load_historical_data(location, days)

        if df.empty:
            logger.error(f"No data available for {location}")
            return pd.DataFrame(), pd.DataFrame()

        # Clean data
        df = self._clean_data(df)

        # Create target variable (rain probability)
        # For now, use rain chance from API if available, else use precipitation
        if 'rain_probability' in df.columns:
            target = df[['timestamp', 'rain_probability']].copy()
            target.columns = ['timestamp', 'target']
        else:
            # Threshold: > 1mm = rain
            target = df[['timestamp', 'precipitation']].copy()
            target.columns = ['timestamp', 'target']
            target['target'] = (target['target'] > 1.0).astype(int)

        features = df[['timestamp', 'temperature', 'humidity', 'wind_speed',
                      'clouds', 'pressure']].copy()

        return features, target

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess data."""
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

        # Remove outliers (simple approach)
        for col in ['temperature', 'humidity', 'wind_speed']:
            if col in df.columns:
                q1 = df[col].quantile(0.01)
                q99 = df[col].quantile(0.99)
                df[col] = df[col].clip(lower=q1, upper=q99)

        return df

    def get_latest_data(self, location: str) -> Optional[Dict]:
        """Get most recent weather data for a location."""
        if location in self.current_data:
            return self.current_data[location]

        # Try loading from file
        pattern = f"{location.lower().replace(' ', '_')}_*.jsonl"
        files = list(self.raw_data_dir.glob(pattern))

        if not files:
            return None

        try:
            with open(sorted(files)[-1], 'r') as f:
                lines = f.readlines()
                if lines:
                    return json.loads(lines[-1])
        except Exception as e:
            logger.error(f"Error loading latest data: {e}")

        return None

    def export_data(self, location: str, output_format: str = 'csv') -> str:
        """
        Export weather data for a location.

        Args:
            location: Location name
            output_format: 'csv' or 'json'

        Returns:
            Path to exported file
        """
        df = self.load_historical_data(location)

        if df.empty:
            logger.error(f"No data to export for {location}")
            return ""

        filename = self.processed_data_dir / f"{location.lower().replace(' ', '_')}_export"

        try:
            if output_format == 'csv':
                filepath = f"{filename}.csv"
                df.to_csv(filepath, index=False)
            elif output_format == 'json':
                filepath = f"{filename}.json"
                df.to_json(filepath, orient='records', date_format='iso')
            else:
                logger.error(f"Unknown format: {output_format}")
                return ""

            logger.info(f"Data exported to {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return ""

    def get_data_stats(self, location: str) -> Dict[str, Any]:
        """Get statistics about available data for a location."""
        df = self.load_historical_data(location)

        if df.empty:
            return {"error": "No data available"}

        stats = {
            "location": location,
            "records": len(df),
            "date_range": {
                "start": df['timestamp'].min().isoformat() if 'timestamp' in df else None,
                "end": df['timestamp'].max().isoformat() if 'timestamp' in df else None,
            },
            "temperature": {
                "min": df['temperature'].min() if 'temperature' in df else None,
                "max": df['temperature'].max() if 'temperature' in df else None,
                "mean": df['temperature'].mean() if 'temperature' in df else None,
            },
            "humidity": {
                "min": df['humidity'].min() if 'humidity' in df else None,
                "max": df['humidity'].max() if 'humidity' in df else None,
                "mean": df['humidity'].mean() if 'humidity' in df else None,
            },
            "wind_speed": {
                "min": df['wind_speed'].min() if 'wind_speed' in df else None,
                "max": df['wind_speed'].max() if 'wind_speed' in df else None,
                "mean": df['wind_speed'].mean() if 'wind_speed' in df else None,
            },
        }

        return stats
