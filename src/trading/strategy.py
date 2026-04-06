"""Trading strategy module."""

from typing import Dict, List, Optional, Any
from datetime import datetime
from src.utils.logger import get_logger
from src.utils.helpers import calculate_kelly_size

logger = get_logger(__name__)


class TradingStrategy:
    """Define and manage trading strategies."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.trading_config = config.get('trading', {})
        self.edge_threshold = self.trading_config.get('edge_threshold', 0.05)
        self.position_sizing = self.trading_config.get('position_sizing', {})

    def evaluate_opportunity(self, opportunity: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Evaluate if a trading opportunity meets strategy criteria.

        Args:
            opportunity: Market opportunity dict

        Returns:
            Trade recommendation or None
        """
        edge = opportunity.get('edge', 0)
        liquidity = opportunity.get('liquidity_usd', 0)
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
            implied_prob = opportunity['predicted_yes_probability']
            market_price = opportunity['current_yes_price']
        elif edge < -self.edge_threshold:
            action = 'BUY_NO'
            implied_prob = opportunity['predicted_no_probability']
            market_price = opportunity['current_no_price']
        else:
            return None

        # Calculate position size
        position_size = self._calculate_position_size(implied_prob, market_price, opportunity)

        return {
            'market_id': opportunity['market_id'],
            'question': opportunity['question'],
            'action': action,
            'position_size': position_size,
            'entry_price': market_price,
            'implied_probability': implied_prob,
            'edge': edge,
            'profit_target': self.trading_config.get('exit_conditions', [{}])[0].get('profit_target', 0.20),
            'stop_loss': self.trading_config.get('exit_conditions', [{}])[0].get('stop_loss', -0.10),
            'max_hold_time': self.trading_config.get('exit_conditions', [{}])[0].get('time_based', 86400),
            'entry_time': datetime.utcnow().isoformat(),
            'status': 'PENDING'
        }

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
        # Score each opportunity
        scored = []
        for opp in opportunities:
            score = self._score_opportunity(opp)
            scored.append((opp, score))

        # Sort by score (highest first)
        scored.sort(key=lambda x: x[1], reverse=True)
        return [opp for opp, _ in scored]

    def _score_opportunity(self, opportunity: Dict) -> float:
        """Score an opportunity (0-100)."""
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
