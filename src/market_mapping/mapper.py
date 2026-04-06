"""Market mapping module - maps weather predictions to Polymarket questions."""

import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd

from src.utils.logger import get_logger
from src.utils.helpers import load_markets

logger = get_logger(__name__)


class MarketMapper:
    """Map weather predictions to Polymarket markets."""

    def __init__(self, markets_file: str = 'config/markets.json'):
        """
        Initialize mapper.

        Args:
            markets_file: Path to markets configuration file
        """
        self.markets_data = load_markets(markets_file)
        self.markets = self.markets_data.get('markets', [])
        self.market_types = self.markets_data.get('market_types', {})
        self.locations = self.markets_data.get('locations', {})

    def get_markets_by_location(self, location: str) -> List[Dict]:
        """Get all markets for a specific location."""
        return [m for m in self.markets if m.get('location') == location and m.get('enabled')]

    def get_markets_by_type(self, prediction_type: str) -> List[Dict]:
        """Get all markets of a specific prediction type."""
        return [m for m in self.markets
                if m.get('prediction_type') == prediction_type and m.get('enabled')]

    def get_active_markets(self) -> List[Dict]:
        """Get all active markets."""
        return [m for m in self.markets if m.get('enabled')]

    def map_prediction_to_market(self, location: str, prediction: Dict[str, Any]) -> List[Dict]:
        """
        Map a weather prediction to matching markets.

        Args:
            location: Weather location
            prediction: Prediction dict with probabilities

        Returns:
            List of market trade recommendations
        """
        if not isinstance(prediction, dict):
            logger.error(f"Prediction must be dict, got {type(prediction)}")
            return []

        markets = self.get_markets_by_location(location)
        recommendations = []

        for market in markets:
            rec = self._create_recommendation(market, prediction)
            if rec and isinstance(rec, dict):
                recommendations.append(rec)
            elif rec:
                logger.warning(f"Recommendation is not dict: {type(rec)}")

        return recommendations

    def _create_recommendation(self, market: Dict, prediction: Dict) -> Optional[Dict]:
        """Create a trade recommendation for a market based on prediction."""
        market_type = market.get('prediction_type')
        yes_prob = self._get_prediction_probability(prediction, market_type)

        if yes_prob is None:
            return None

        current_yes_price = market.get('current_price_yes', 0.5)
        edge = yes_prob - current_yes_price

        return {
            'market_id': market.get('id'),
            'question': market.get('question'),
            'location': market.get('location'),
            'prediction_type': market_type,
            'predicted_yes_probability': float(yes_prob),
            'predicted_no_probability': float(1 - yes_prob),
            'current_yes_price': float(current_yes_price),
            'current_no_price': float(market.get('current_price_no', 0.5)),
            'edge': float(edge),
            'liquidity_usd': market.get('liquidity_usd'),
            'volume_24h_usd': market.get('volume_24h_usd'),
            'resolution_date': market.get('resolution_date'),
            'recommendation': self._get_recommendation(edge, market),
            'timestamp': datetime.utcnow().isoformat()
        }

    def _get_prediction_probability(self, prediction: Dict, market_type: str) -> Optional[float]:
        """Extract relevant probability from prediction for market type."""
        if market_type == 'rain_probability':
            return prediction.get('rain_probability')
        elif market_type == 'temperature_threshold':
            # Would need temperature prediction
            return None
        elif market_type == 'wind_threshold':
            return prediction.get('wind_probability')
        elif market_type == 'humidity_threshold':
            return prediction.get('humidity_probability')
        elif market_type == 'snow_probability':
            return prediction.get('snow_probability')

        return None

    def _get_recommendation(self, edge: float, market: Dict) -> str:
        """Get trade recommendation based on edge."""
        edge_threshold = 0.05  # 5% default

        if edge > edge_threshold:
            return 'BUY_YES'
        elif edge < -edge_threshold:
            return 'BUY_NO'
        else:
            return 'SKIP'

    def find_opportunities(self, predictions: Dict[str, Dict],
                         edge_threshold: float = 0.05) -> List[Dict]:
        """
        Find trading opportunities across all markets.

        Args:
            predictions: Dict mapping locations to predictions
            edge_threshold: Minimum probability edge required

        Returns:
            List of opportunities sorted by edge size
        """
        opportunities = []

        if not isinstance(predictions, dict):
            logger.error(f"Predictions must be dict, got {type(predictions)}")
            return []

        for location, prediction in predictions.items():
            try:
                recommendations = self.map_prediction_to_market(location, prediction)

                # Defensive: ensure recommendations is a list
                if not isinstance(recommendations, list):
                    logger.warning(f"map_prediction_to_market returned {type(recommendations)} instead of list")
                    recommendations = []

                for rec in recommendations:
                    if isinstance(rec, dict) and abs(rec.get('edge', 0)) > edge_threshold:
                        opportunities.append(rec)

            except Exception as e:
                logger.warning(f"Error processing opportunities for {location}: {e}")
                continue

        # Sort by edge size (largest opportunities first)
        try:
            return sorted(opportunities, key=lambda x: abs(x.get('edge', 0)), reverse=True)
        except Exception as e:
            logger.error(f"Error sorting opportunities: {e}")
            return opportunities

    def validate_market_mapping(self) -> Dict[str, Any]:
        """Validate market configurations."""
        validation = {
            'total_markets': len(self.markets),
            'enabled_markets': sum(1 for m in self.markets if m.get('enabled')),
            'by_location': {},
            'by_type': {},
            'issues': []
        }

        # Check by location
        for location in self.locations:
            count = len(self.get_markets_by_location(location))
            validation['by_location'][location] = count

        # Check by type
        for mtype in self.market_types:
            count = len(self.get_markets_by_type(mtype))
            validation['by_type'][mtype] = count

        # Validate each market
        for market in self.markets:
            # Check required fields
            required_fields = ['id', 'question', 'location', 'prediction_type']
            for field in required_fields:
                if field not in market:
                    validation['issues'].append(
                        f"Market {market.get('id', 'unknown')} missing field: {field}"
                    )

            # Check if location exists
            if market.get('location') not in self.locations:
                validation['issues'].append(
                    f"Market {market.get('id')} references unknown location: {market.get('location')}"
                )

            # Check if type is valid
            if market.get('prediction_type') not in self.market_types:
                validation['issues'].append(
                    f"Market {market.get('id')} references unknown type: {market.get('prediction_type')}"
                )

        return validation

    def update_market_prices(self, price_updates: Dict[str, Dict]) -> None:
        """
        Update current market prices.

        Args:
            price_updates: Dict mapping market IDs to price updates
        """
        for market in self.markets:
            market_id = market.get('id')
            if market_id in price_updates:
                update = price_updates[market_id]
                market['current_price_yes'] = update.get('yes', market.get('current_price_yes'))
                market['current_price_no'] = update.get('no', market.get('current_price_no'))
                market['liquidity_usd'] = update.get('liquidity', market.get('liquidity_usd'))

    def export_mapped_data(self, predictions: Dict[str, Dict], output_file: str) -> bool:
        """
        Export predictions with market mappings.

        Args:
            predictions: Predictions by location
            output_file: Output file path

        Returns:
            True if successful
        """
        try:
            opportunities = self.find_opportunities(predictions)

            export_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'opportunities': opportunities,
                'predictions_by_location': predictions,
                'market_stats': self.validate_market_mapping()
            }

            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

            logger.info(f"Exported {len(opportunities)} opportunities to {output_file}")
            return True

        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return False

    def get_market_summary(self) -> pd.DataFrame:
        """Get summary statistics of all markets."""
        df = pd.DataFrame(self.markets)

        if df.empty:
            return df

        # Select relevant columns
        summary_cols = ['id', 'question', 'location', 'prediction_type',
                       'current_price_yes', 'current_price_no',
                       'liquidity_usd', 'volume_24h_usd', 'enabled']

        available_cols = [c for c in summary_cols if c in df.columns]
        return df[available_cols]
