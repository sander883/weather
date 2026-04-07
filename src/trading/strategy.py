"""Trading strategy module."""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from src.utils.logger import get_logger
from src.utils.helpers import calculate_kelly_size
from src.models.schemas import MarketOpportunity, Trade, TradingConfig

logger = get_logger(__name__)


class TradingStrategy:
    """Define and manage trading strategies."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.trading_config = config.get('trading', {})
        self.edge_threshold = self.trading_config.get('edge_threshold', 0.05)
        self.position_sizing = self.trading_config.get('position_sizing', {})

    def evaluate_opportunity(self, opportunity: Union[Dict[str, Any], MarketOpportunity]) -> Optional[Union[Dict[str, Any], Trade]]:
        """
        Evaluate if a trading opportunity meets strategy criteria.

        Args:
            opportunity: Market opportunity dict or MarketOpportunity model

        Returns:
            Trade recommendation dict or Trade model, or None
        """
        # Convert Pydantic model to dict if needed
        opp_dict = opportunity.model_dump() if isinstance(opportunity, MarketOpportunity) else opportunity
        return_model = isinstance(opportunity, MarketOpportunity)

        # Defensive: ensure opportunity is a dict
        if not isinstance(opp_dict, dict):
            logger.warning(f"Opportunity is not a dict: {type(opp_dict)}")
            return None

        try:
            edge = opp_dict.get('edge', 0)
            liquidity = opp_dict.get('liquidity_usd', 0)
            min_liquidity = self.trading_config.get('entry_conditions', [{}])[0].get('min_liquidity', 500)

            # Check minimum edge
            if abs(edge) < self.edge_threshold:
                logger.debug(f"Edge {edge} below threshold {self.edge_threshold}")
                return None

            # Check minimum liquidity
            if liquidity < min_liquidity:
                logger.warning(f"Liquidity {liquidity} below minimum {min_liquidity}")
                return None

            # Determine trade direction
            if edge > self.edge_threshold:
                action = 'BUY_YES'
                implied_prob = opp_dict.get('predicted_yes_probability', 0.5)
                market_price = opp_dict.get('current_yes_price', 0.5)
            elif edge < -self.edge_threshold:
                action = 'BUY_NO'
                implied_prob = opp_dict.get('predicted_no_probability', 0.5)
                market_price = opp_dict.get('current_no_price', 0.5)
            else:
                return None

            # Calculate position size
            position_size = self._calculate_position_size(implied_prob, market_price, opp_dict)

            trade_dict = {
                'market_id': opp_dict.get('market_id'),
                'question': opp_dict.get('question'),
                'location': opp_dict.get('location'),
                'action': action,
                'position_size': position_size,
                'entry_price': market_price,
                'implied_probability': implied_prob,
                'edge': edge,
                'profit_target': self.trading_config.get('exit_conditions', [{}])[0].get('profit_target', 0.20),
                'stop_loss': self.trading_config.get('exit_conditions', [{}])[0].get('stop_loss', -0.10),
                'max_hold_time': self.trading_config.get('exit_conditions', [{}])[0].get('time_based', 86400),
                'entry_time': datetime.utcnow(),
                'status': 'PENDING'
            }

            # Return as Pydantic model if input was model
            if return_model:
                return Trade(**trade_dict)
            return trade_dict

        except Exception as e:
            logger.error(f"Error evaluating opportunity: {e}")
            return None

    def _calculate_position_size(self, prob: float, price: float,
                                 opportunity: Dict) -> float:
        """
        Calculate position size using configured method.

        Args:
            prob: Predicted probability
            price: Market price
            opportunity: Opportunity dict with liquidity info

        Returns:
            Position size in USD
        """
        method = self.position_sizing.get('method', 'kelly')
        capital = self.config.get('risk_management', {}).get('portfolio', {}).get('initial_capital', 10000)

        max_size = self.trading_config.get('max_position_size', 1000)
        min_size = self.trading_config.get('min_position_size', 10)

        liquidity = opportunity.get('liquidity_usd', 10000)

        if method == 'kelly':
            # Kelly Criterion position sizing
            win_loss_ratio = 1.0  # Assume equal win/loss on binary options
            kelly_fraction = calculate_kelly_size(prob, win_loss_ratio)
            kelly_adjusted = kelly_fraction * self.position_sizing.get('kelly_fraction', 0.25)

            # Risk per trade
            risk_per_trade = capital * kelly_adjusted
            position_size = min(risk_per_trade, liquidity * 0.1)  # Max 10% of market liquidity

        elif method == 'fixed':
            # Fixed percentage of capital
            percent = self.position_sizing.get('fixed_size_percent', 2.0) / 100
            position_size = capital * percent

        else:
            position_size = capital * 0.02  # Default 2%

        # Clamp to min/max
        position_size = max(min_size, min(position_size, max_size))

        logger.info(f"Position size: {position_size} USD (method: {method})")
        return position_size

    def should_exit_position(self, position: Dict[str, Any]) -> Optional[str]:
        """
        Determine if a position should be exited.

        Args:
            position: Open position dict

        Returns:
            Exit reason or None
        """
        exit_conditions = self.trading_config.get('exit_conditions', [{}])[0]

        # Profit target
        if position.get('current_pnl', 0) >= exit_conditions.get('profit_target', 0.20):
            return 'PROFIT_TARGET'

        # Stop loss
        if position.get('current_pnl', 0) <= exit_conditions.get('stop_loss', -0.10):
            return 'STOP_LOSS'

        # Time-based exit
        entry_time_str = position.get('entry_time') or position.get('execution_time')
        if not entry_time_str:
            return None

        entry_time = datetime.fromisoformat(entry_time_str)
        max_hold = exit_conditions.get('time_based', 86400)
        elapsed = (datetime.utcnow() - entry_time).total_seconds()

        if elapsed > max_hold:
            return 'TIME_LIMIT'

        return None

    def rank_opportunities(self, opportunities: List[Dict]) -> List[Dict]:
        """
        Rank opportunities by attractiveness.

        Args:
            opportunities: List of opportunity dicts

        Returns:
            Sorted list of opportunities
        """
        # Defensive: ensure opportunities is a list
        if not isinstance(opportunities, list):
            logger.warning(f"Opportunities must be list, got {type(opportunities)}")
            return []

        # Defensive: filter out non-dicts
        valid_opps = [opp for opp in opportunities if isinstance(opp, dict)]
        if len(valid_opps) < len(opportunities):
            logger.warning(f"Filtered {len(opportunities) - len(valid_opps)} non-dict opportunities")

        # Score each opportunity
        scored = []
        for opp in valid_opps:
            try:
                score = self._score_opportunity(opp)
                scored.append((opp, score))
            except Exception as e:
                logger.warning(f"Error scoring opportunity: {e}")
                continue

        # Sort by score (highest first)
        scored.sort(key=lambda x: x[1], reverse=True)
        return [opp for opp, _ in scored]

    def _score_opportunity(self, opportunity: Dict) -> float:
        """Score an opportunity (0-100)."""
        # Defensive: ensure opportunity is a dict
        if not isinstance(opportunity, dict):
            logger.warning(f"Opportunity must be dict, got {type(opportunity)}")
            return 0.0

        try:
            edge = abs(opportunity.get('edge', 0))
            liquidity = opportunity.get('liquidity_usd', 0)
            volume = opportunity.get('volume_24h_usd', 0)

            # Edge score (0-50)
            edge_score = min(50, edge * 100)

            # Liquidity score (0-30)
            liquidity_score = min(30, (liquidity / 1000))

            # Volume score (0-20)
            volume_score = min(20, (volume / 500))

            total_score = edge_score + liquidity_score + volume_score
            return total_score

        except Exception as e:
            logger.error(f"Error calculating opportunity score: {e}")
            return 0.0

    def get_portfolio_allocation(self, opportunities: List[Dict],
                                capital: float = 10000) -> List[Dict]:
        """
        Allocate capital across multiple opportunities.

        Args:
            opportunities: Ranked list of opportunities
            capital: Total capital to allocate

        Returns:
            List of allocations with sizes
        """
        max_positions = self.trading_config.get('max_concurrent_positions', 5)
        max_concentration = self.config.get('risk_management', {}).get('position', {}).get('max_concentration', 0.30)

        allocations = []
        remaining_capital = capital
        max_per_trade = capital * max_concentration

        for opp in opportunities[:max_positions]:
            if remaining_capital <= 0:
                break

            # Get position size (limited by capital and concentration)
            position_size = opp.get('position_size', capital * 0.02)
            position_size = min(position_size, remaining_capital, max_per_trade)

            if position_size >= self.trading_config.get('min_position_size', 10):
                allocation = opp.copy()
                allocation['position_size'] = position_size
                allocations.append(allocation)
                remaining_capital -= position_size

        logger.info(f"Allocated {capital - remaining_capital} across {len(allocations)} positions")
        return allocations
