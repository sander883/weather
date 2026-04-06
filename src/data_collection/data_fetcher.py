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
        """Save raw weather data to file with all required fields."""
        try:
            timestamp = datetime.now().isoformat()
            filename = self.raw_data_dir / f"{location.lower().replace(' ', '_')}_{timestamp.split('T')[0]}.jsonl"

            # Ensure ALL required fields exist with sensible defaults
            data_with_defaults = {
                'timestamp': data.get('timestamp', timestamp),
                'temperature': float(data.get('temperature') or 15),
                'humidity': float(data.get('humidity') or 50),
                'wind_speed': float(data.get('wind_speed') or 0),
                'clouds': float(data.get('clouds') or 50),
                'precipitation': float(data.get('precipitation') or data.get('precip_mm') or 0),
                'pressure': float(data.get('pressure') or data.get('pressure_mb') or 1013.25),
                'rain_probability': float(data.get('rain_probability') or data.get('rain_chance') or data.get('chance_of_rain', 0)) / 100.0
                    if data.get('rain_probability') or data.get('rain_chance') or data.get('chance_of_rain')
                    else 0.0,
                **data  # Include all original fields
            }

            with open(filename, 'a') as f:
                data_with_defaults['saved_at'] = timestamp
                f.write(json.dumps(data_with_defaults, default=str) + '\n')
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
        Prepare data for model training with proper alignment.

        Returns:
            Tuple of (features_df, target_series)
        """
        df = self.load_historical_data(location, days)

        if df.empty:
            logger.error(f"No data available for {location}")
            return pd.DataFrame(), pd.Series(dtype=float)

        # Ensure timestamp is datetime and sort
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
        else:
            logger.warning(f"No timestamp column in data for {location}")
            df = df.reset_index(drop=True)

        # Clean data first
        df = self._clean_data(df)

        # Remove rows with any NaN in key columns
        key_cols = ['temperature', 'humidity', 'wind_speed', 'clouds', 'precipitation']
        df = df.dropna(subset=key_cols)

        if df.empty:
            logger.error(f"No valid data after cleaning for {location}")
            return pd.DataFrame(), pd.Series(dtype=float)

        # Create target variable
        if 'rain_probability' in df.columns and df['rain_probability'].notna().any():
            target = df['rain_probability'].copy()
        else:
            # Threshold: > 1mm = rain
            target = (df['precipitation'] > 1.0).astype(float)

        # Select features (only numeric, no timestamp)
        feature_cols = ['temperature', 'humidity', 'wind_speed', 'clouds', 'pressure']
        features = df[feature_cols].copy()

        # Ensure all columns are numeric
        for col in features.columns:
            features[col] = pd.to_numeric(features[col], errors='coerce')

        # Fill any remaining NaN
        features = features.fillna(features.mean())

        # CRITICAL: Ensure alignment
        assert len(features) == len(target), f"Misaligned data: features={len(features)}, target={len(target)}"
        assert features.index.equals(target.index) or (len(features) == len(target)), \
            "Features and target indices don't match"

        logger.info(f"Prepared {len(features)} samples with {len(features.columns)} features for {location}")
        return features, target

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess data."""
        # Ensure all required columns exist
        required_cols = ['temperature', 'humidity', 'wind_speed', 'clouds', 'precipitation', 'pressure']
        for col in required_cols:
            if col not in df.columns:
                df[col] = np.nan

        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Fill missing values column-wise with column mean, or global mean if all NaN
        for col in numeric_cols:
            if df[col].isna().all():
                # If entire column is NaN, use a default value
                if col == 'pressure':
                    df[col] = 1013.25  # Standard atmospheric pressure
                elif col == 'humidity':
                    df[col] = 50
                elif col == 'temperature':
                    df[col] = 15
                elif col == 'wind_speed':
                    df[col] = 0
                elif col == 'clouds':
                    df[col] = 50
                elif col == 'precipitation':
                    df[col] = 0
                else:
                    df[col] = df[col].mean() or 0
            else:
                # Fill with column mean
                df[col] = df[col].fillna(df[col].mean())

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
