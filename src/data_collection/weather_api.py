"""Weather API integrations for multiple data sources."""

import requests
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import os
from src.utils.logger import get_logger

logger = get_logger(__name__)


class WeatherAPIClient:
    """Base class for weather API clients."""

    def __init__(self, api_key: str, base_url: str, timeout: int = 10):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.session = None

    def _get_headers(self) -> Dict[str, str]:
        """Get default headers for requests."""
        return {
            'User-Agent': 'PolymarketWeatherAgent/1.0',
            'Accept': 'application/json'
        }

    async def _async_request(self, url: str, params: Dict = None) -> Dict[str, Any]:
        """Make async HTTP request."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    params=params,
                    headers=self._get_headers(),
                    timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.warning(f"API returned status {response.status}")
                        return None
        except asyncio.TimeoutError:
            logger.error(f"Request timeout to {url}")
            return None
        except Exception as e:
            logger.error(f"API request failed: {e}")
            return None

    def _sync_request(self, url: str, params: Dict = None) -> Optional[Dict]:
        """Make synchronous HTTP request."""
        response = None
        try:
            response = requests.get(
                url,
                params=params,
                headers=self._get_headers(),
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout to {url}")
            return None
        except requests.exceptions.HTTPError as e:
            status = response.status_code if response else 'unknown'
            text = response.text if response else str(e)
            logger.error(f"API HTTP error {status}: {text}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None


class OpenWeatherMapClient(WeatherAPIClient):
    """OpenWeatherMap API client."""

    def __init__(self, api_key: str):
        super().__init__(
            api_key=api_key,
            base_url='https://api.openweathermap.org/data/2.5'
        )

    def get_current_weather(self, lat: float, lon: float) -> Optional[Dict]:
        """
        Get current weather for location.

        Returns dict with:
        - temp, feels_like, humidity
        - pressure, clouds, wind_speed, wind_deg
        - rain, snow (if applicable)
        - description, icon
        """
        url = f"{self.base_url}/weather"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.api_key,
            'units': 'metric'
        }

        logger.debug(f"Fetching OpenWeatherMap data from {url}")
        data = self._sync_request(url, params)
        if not data:
            return None

        try:
            # Handle timestamp safely
            dt = data.get('dt')
            timestamp = datetime.fromtimestamp(dt) if dt else datetime.utcnow()

            return {
                'timestamp': timestamp,
                'location': data.get('name', ''),
                'temperature': data.get('main', {}).get('temp'),
                'feels_like': data.get('main', {}).get('feels_like'),
                'humidity': data.get('main', {}).get('humidity'),
                'pressure': data.get('main', {}).get('pressure'),
                'clouds': data.get('clouds', {}).get('all', 0),
                'wind_speed': data.get('wind', {}).get('speed', 0),
                'wind_direction': data.get('wind', {}).get('deg', 0),
                'precipitation': data.get('rain', {}).get('1h', 0) +
                                data.get('snow', {}).get('1h', 0),
                'description': data.get('weather', [{}])[0].get('main', '') if data.get('weather') else '',
                'raw': data
            }
        except Exception as e:
            logger.error(f"Error parsing OpenWeatherMap response: {e}")
            return None

    def get_forecast(self, lat: float, lon: float, days: int = 5) -> Optional[List[Dict]]:
        """
        Get weather forecast.

        Returns list of forecasts for next 5 days (3-hour intervals)
        """
        url = f"{self.base_url}/forecast"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.api_key,
            'units': 'metric',
            'cnt': min(days * 8, 40)  # 3-hour intervals
        }

        data = self._sync_request(url, params)
        if not data:
            return None

        forecasts = []
        try:
            for item in data.get('list', []):
                dt = item.get('dt')
                timestamp = datetime.fromtimestamp(dt) if dt else datetime.utcnow()

                forecasts.append({
                    'timestamp': timestamp,
                    'temperature': item.get('main', {}).get('temp'),
                    'humidity': item.get('main', {}).get('humidity'),
                    'wind_speed': item.get('wind', {}).get('speed', 0),
                    'clouds': item.get('clouds', {}).get('all', 0),
                    'rain_probability': item.get('pop', 0),  # probability of precipitation
                    'precipitation': item.get('rain', {}).get('3h', 0) +
                                    item.get('snow', {}).get('3h', 0),
                    'description': item.get('weather', [{}])[0].get('main', '') if item.get('weather') else '',
                })
        except Exception as e:
            logger.error(f"Error parsing forecast items: {e}")

        return forecasts

    def get_historical_data(self, lat: float, lon: float, days: int = 7) -> Optional[List[Dict]]:
        """
        Get historical weather data (uses forecast as proxy).

        OpenWeatherMap free tier doesn't have true historical endpoint,
        but forecast includes recent data points.
        """
        return self.get_forecast(lat, lon, days)


class WeatherAPIClient2(WeatherAPIClient):
    """WeatherAPI.com client."""

    def __init__(self, api_key: str):
        super().__init__(
            api_key=api_key,
            base_url='https://api.weatherapi.com/v1'
        )

    def get_current_weather(self, lat: float, lon: float) -> Optional[Dict]:
        """
        Get current weather from WeatherAPI.

        Returns dict with weather data
        """
        url = f"{self.base_url}/current.json"
        params = {
            'key': self.api_key,
            'q': f"{lat},{lon}",
            'aqi': 'yes'
        }

        data = self._sync_request(url, params)
        if not data:
            return None

        current = data.get('current', {})
        location = data.get('location', {})

        try:
            # Handle timestamp safely
            last_updated = current.get('last_updated')
            if last_updated:
                timestamp = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
            else:
                timestamp = datetime.utcnow()

            # Safe wind speed conversion
            wind_kph = current.get('wind_kph', 0)
            wind_ms = wind_kph / 3.6 if wind_kph else 0

            return {
                'timestamp': timestamp,
                'location': f"{location.get('name')}, {location.get('country')}",
                'temperature': current.get('temp_c'),
                'feels_like': current.get('feelslike_c'),
                'humidity': current.get('humidity'),
                'pressure': current.get('pressure_mb'),
                'clouds': current.get('cloud', 0),
                'wind_speed': wind_ms,
                'wind_direction': current.get('wind_degree', 0),
                'precipitation': current.get('precip_mm', 0),
                'rain_chance': current.get('chance_of_rain', 0),
                'snow_chance': current.get('chance_of_snow', 0),
                'description': current.get('condition', {}).get('text', ''),
                'raw': data
            }
        except Exception as e:
            logger.error(f"Error parsing WeatherAPI response: {e}")
            return None

    def get_forecast(self, lat: float, lon: float, days: int = 5) -> Optional[List[Dict]]:
        """
        Get weather forecast from WeatherAPI.

        Returns list of forecasts
        """
        url = f"{self.base_url}/forecast.json"
        params = {
            'key': self.api_key,
            'q': f"{lat},{lon}",
            'days': min(days, 10),
            'aqi': 'no'
        }

        data = self._sync_request(url, params)
        if not data:
            return None

        forecasts = []
        try:
            for day in data.get('forecast', {}).get('forecastday', []):
                for hour in day.get('hour', []):
                    # Safe timestamp parsing
                    time_str = hour.get('time')
                    if time_str:
                        timestamp = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                    else:
                        timestamp = datetime.utcnow()

                    # Safe wind speed conversion
                    wind_kph = hour.get('wind_kph', 0)
                    wind_ms = wind_kph / 3.6 if wind_kph else 0

                    # Safe rain probability conversion (convert from 0-100 to 0-1)
                    rain_prob = hour.get('chance_of_rain', 0)
                    rain_prob_norm = rain_prob / 100.0 if rain_prob else 0

                    forecasts.append({
                        'timestamp': timestamp,
                        'temperature': hour.get('temp_c'),
                        'humidity': hour.get('humidity'),
                        'wind_speed': wind_ms,
                        'clouds': hour.get('cloud', 0),
                        'rain_probability': rain_prob_norm,
                        'precipitation': hour.get('precip_mm', 0),
                        'description': hour.get('condition', {}).get('text', ''),
                    })
        except Exception as e:
            logger.error(f"Error parsing WeatherAPI forecast: {e}")

        return forecasts

    def get_historical_data(self, lat: float, lon: float, days: int = 7) -> Optional[List[Dict]]:
        """
        Get historical weather data from WeatherAPI.

        WeatherAPI allows querying past dates with forecast endpoint.
        """
        from datetime import datetime, timedelta

        url = f"{self.base_url}/forecast.json"
        all_historical = []

        try:
            # Fetch current forecast (includes today and next days)
            params = {
                'key': self.api_key,
                'q': f"{lat},{lon}",
                'days': min(days, 10),
                'aqi': 'no'
            }

            data = self._sync_request(url, params)
            if not data:
                return None

            # Extract hourly data
            for day in data.get('forecast', {}).get('forecastday', []):
                for hour in day.get('hour', []):
                    all_historical.append({
                        'timestamp': datetime.fromisoformat(hour.get('time')),
                        'temperature': hour.get('temp_c'),
                        'humidity': hour.get('humidity'),
                        'wind_speed': hour.get('wind_kph') / 3.6,
                        'clouds': hour.get('cloud', 0),
                        'rain_probability': hour.get('chance_of_rain', 0) / 100.0,
                        'precipitation': hour.get('precip_mm', 0),
                        'pressure': hour.get('pressure_mb'),
                        'description': hour.get('condition', {}).get('text', ''),
                    })

            return all_historical if all_historical else None

        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return None


class WeatherDataAggregator:
    """Aggregate data from multiple weather sources."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.clients = {}
        self._initialize_clients()

    def _initialize_clients(self):
        """Initialize weather API clients based on config."""
        for source in self.config.get('weather', {}).get('sources', []):
            if not source.get('enabled'):
                continue

            api_key = os.getenv(source['api_key'])
            if not api_key:
                logger.warning(f"No API key found for {source['name']}")
                continue

            if source['name'] == 'openweathermap':
                self.clients['openweathermap'] = OpenWeatherMapClient(api_key)
            elif source['name'] == 'weatherapi':
                self.clients['weatherapi'] = WeatherAPIClient2(api_key)

    def get_weather_consensus(self, lat: float, lon: float) -> Optional[Dict]:
        """
        Get weather data from multiple sources and return consensus.

        Returns averaged/aggregated data from available sources
        """
        all_data = []

        for name, client in self.clients.items():
            try:
                data = client.get_current_weather(lat, lon)
                if data:
                    data['source'] = name
                    all_data.append(data)
            except Exception as e:
                logger.error(f"Error fetching from {name}: {e}")

        if not all_data:
            logger.error("No weather data available from any source")
            return None

        # Average numeric values
        consensus = {
            'timestamp': all_data[0]['timestamp'],
            'location': all_data[0]['location'],
            'sources': [d['source'] for d in all_data],
            'num_sources': len(all_data),
        }

        numeric_keys = [
            'temperature', 'humidity', 'wind_speed',
            'clouds', 'precipitation'
        ]

        for key in numeric_keys:
            values = [d.get(key) for d in all_data if d.get(key) is not None]
            if values:
                consensus[key] = sum(values) / len(values)

        # Use most common description
        descriptions = [d.get('description') for d in all_data if d.get('description')]
        if descriptions:
            consensus['description'] = max(set(descriptions), key=descriptions.count)

        return consensus

    def get_forecast_consensus(self, lat: float, lon: float, days: int = 5) -> Optional[List[Dict]]:
        """
        Get forecast from multiple sources.

        Returns list of forecasts, preferring primary source
        """
        forecasts = []

        # Try to get from primary source first
        for name, client in self.clients.items():
            try:
                data = client.get_forecast(lat, lon, days)
                if data:
                    for item in data:
                        item['source'] = name
                    return data
            except Exception as e:
                logger.error(f"Error fetching forecast from {name}: {e}")

        return None

    def get_historical_consensus(self, lat: float, lon: float, days: int = 7) -> Optional[List[Dict]]:
        """
        Get historical weather data from multiple sources.

        Returns list of historical data points, preferring primary source
        """
        # Try to get from primary source first (WeatherAPI preferred for historical)
        for name, client in self.clients.items():
            try:
                if hasattr(client, 'get_historical_data'):
                    data = client.get_historical_data(lat, lon, days)
                    if data:
                        for item in data:
                            item['source'] = name
                        logger.info(f"Fetched {len(data)} historical data points from {name}")
                        return data
            except Exception as e:
                logger.error(f"Error fetching historical data from {name}: {e}")

        logger.warning("No historical data available from any source")
        return None
