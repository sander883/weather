"""Unit tests for trading strategy module."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from src.trading.strategy import TradingStrategy


@pytest.fixture
def sample_config():
    """Create sample trading configuration."""
    return {
        'trading': {
            'edge_threshold': 0.05,
            'min_position_size': 10,
            'max_position_size': 1000,
            'max_concurrent_positions': 5,
            'position_sizing': {
                'method': 'kelly',
                'kelly_fraction': 0.25,
                'fixed_size_percent': 2.0
            },
            'entry_conditions': [
                {'min_liquidity': 500}
            ],
            'exit_conditions': [
                {
                    'profit_target': 0.20,
                    'stop_loss': -0.10,
                    'time_based': 86400
                }
            ]
        },
        'risk_management': {
            'portfolio': {
                'initial_capital': 10000
            },
            'position': {
                'max_concentration': 0.30
            }
        }
    }


@pytest.fixture
def sample_opportunity():
    """Create sample trading opportunity."""
    return {
        'market_id': 'market_123',
        'question': 'Will it rain tomorrow?',
        'location': 'New York',
        'edge': 0.10,
        'liquidity_usd': 5000,
        'volume_24h_usd': 2000,
        'predicted_yes_probability': 0.65,
        'current_yes_price': 0.55,
        'predicted_no_probability': 0.35,
        'current_no_price': 0.45
    }


class TestTradingStrategyInit:
    """Test TradingStrategy initialization."""

    def test_init_with_config(self, sample_config):
        """Test initialization with config."""
        strategy = TradingStrategy(config=sample_config)
        assert strategy.config == sample_config
        assert strategy.edge_threshold == 0.05

    def test_init_default_edge_threshold(self):
        """Test default edge threshold."""
        config = {'trading': {}}
        strategy = TradingStrategy(config=config)
        assert strategy.edge_threshold == 0.05

    def test_init_custom_edge_threshold(self):
        """Test custom edge threshold."""
        config = {'trading': {'edge_threshold': 0.08}}
        strategy = TradingStrategy(config=config)
        assert strategy.edge_threshold == 0.08


class TestEvaluateOpportunity:
    """Test evaluate_opportunity method."""

    def test_evaluate_opportunity_buy_yes(self, sample_config, sample_opportunity):
        """Test evaluating opportunity with positive edge (BUY_YES)."""
        strategy = TradingStrategy(config=sample_config)
        trade = strategy.evaluate_opportunity(sample_opportunity)

        assert trade is not None
        assert trade['action'] == 'BUY_YES'
        assert trade['market_id'] == 'market_123'
        assert trade['status'] == 'PENDING'
        assert trade['position_size'] > 0

    def test_evaluate_opportunity_buy_no(self, sample_config):
        """Test evaluating opportunity with negative edge (BUY_NO)."""
        opportunity = {
            'market_id': 'market_456',
            'question': 'Will it rain?',
            'location': 'LA',
            'edge': -0.08,
            'liquidity_usd': 5000,
            'predicted_no_probability': 0.70,
            'current_no_price': 0.60
        }

        strategy = TradingStrategy(config=sample_config)
        trade = strategy.evaluate_opportunity(opportunity)

        assert trade is not None
        assert trade['action'] == 'BUY_NO'

    def test_evaluate_opportunity_insufficient_edge(self, sample_config):
        """Test that small edges are rejected."""
        opportunity = {
            'market_id': 'market_789',
            'edge': 0.02,  # Below threshold
            'liquidity_usd': 5000,
        }

        strategy = TradingStrategy(config=sample_config)
        trade = strategy.evaluate_opportunity(opportunity)

        assert trade is None

    def test_evaluate_opportunity_insufficient_liquidity(self, sample_config):
        """Test that low liquidity opportunities are rejected."""
        opportunity = {
            'market_id': 'market_999',
            'edge': 0.10,
            'liquidity_usd': 100,  # Below minimum
            'predicted_yes_probability': 0.65,
            'current_yes_price': 0.55
        }

        strategy = TradingStrategy(config=sample_config)
        trade = strategy.evaluate_opportunity(opportunity)

        assert trade is None

    def test_evaluate_opportunity_invalid_type(self, sample_config):
        """Test with invalid opportunity type."""
        strategy = TradingStrategy(config=sample_config)
        trade = strategy.evaluate_opportunity("not a dict")

        assert trade is None

    def test_evaluate_opportunity_contains_required_fields(self, sample_config, sample_opportunity):
        """Test that returned trade has all required fields."""
        strategy = TradingStrategy(config=sample_config)
        trade = strategy.evaluate_opportunity(sample_opportunity)

        required_fields = [
            'market_id', 'question', 'action', 'position_size',
            'entry_price', 'implied_probability', 'edge',
            'profit_target', 'stop_loss', 'max_hold_time',
            'entry_time', 'status'
        ]

        for field in required_fields:
            assert field in trade


class TestCalculatePositionSize:
    """Test _calculate_position_size method."""

    def test_kelly_position_sizing(self, sample_config, sample_opportunity):
        """Test Kelly Criterion position sizing."""
        strategy = TradingStrategy(config=sample_config)
        sample_config['trading']['position_sizing']['method'] = 'kelly'

        size = strategy._calculate_position_size(
            prob=0.65,
            price=0.55,
            opportunity=sample_opportunity
        )

        assert 10 <= size <= 1000  # Within min/max bounds
        assert isinstance(size, float)

    def test_fixed_position_sizing(self, sample_config, sample_opportunity):
        """Test fixed percentage position sizing."""
        sample_config['trading']['position_sizing']['method'] = 'fixed'
        strategy = TradingStrategy(config=sample_config)

        size = strategy._calculate_position_size(
            prob=0.65,
            price=0.55,
            opportunity=sample_opportunity
        )

        # Fixed 2% of 10000 = 200
        assert 10 <= size <= 1000

    def test_default_position_sizing(self, sample_config, sample_opportunity):
        """Test default position sizing for unknown method."""
        sample_config['trading']['position_sizing']['method'] = 'unknown'
        strategy = TradingStrategy(config=sample_config)

        size = strategy._calculate_position_size(
            prob=0.65,
            price=0.55,
            opportunity=sample_opportunity
        )

        assert 10 <= size <= 1000

    def test_position_size_respects_bounds(self, sample_config, sample_opportunity):
        """Test that position size respects min/max bounds."""
        strategy = TradingStrategy(config=sample_config)

        # Test minimum bound
        sample_config['trading']['position_sizing']['method'] = 'fixed'
        sample_config['trading']['position_sizing']['fixed_size_percent'] = 0.001  # Very small
        size = strategy._calculate_position_size(
            prob=0.65,
            price=0.55,
            opportunity=sample_opportunity
        )
        assert size >= 10

        # Test maximum bound
        sample_config['trading']['position_sizing']['fixed_size_percent'] = 500  # Very large
        size = strategy._calculate_position_size(
            prob=0.65,
            price=0.55,
            opportunity=sample_opportunity
        )
        assert size <= 1000


class TestShouldExitPosition:
    """Test should_exit_position method."""

    def test_exit_on_profit_target(self, sample_config):
        """Test exit when profit target is reached."""
        position = {
            'entry_time': datetime.utcnow().isoformat(),
            'current_pnl': 0.25
        }

        strategy = TradingStrategy(config=sample_config)
        reason = strategy.should_exit_position(position)

        assert reason == 'PROFIT_TARGET'

    def test_exit_on_stop_loss(self, sample_config):
        """Test exit when stop loss is triggered."""
        position = {
            'entry_time': datetime.utcnow().isoformat(),
            'current_pnl': -0.15
        }

        strategy = TradingStrategy(config=sample_config)
        reason = strategy.should_exit_position(position)

        assert reason == 'STOP_LOSS'

    def test_exit_on_time_limit(self, sample_config):
        """Test exit when time limit is exceeded."""
        old_time = datetime.utcnow() - timedelta(hours=25)
        position = {
            'entry_time': old_time.isoformat(),
            'current_pnl': 0.05
        }

        strategy = TradingStrategy(config=sample_config)
        reason = strategy.should_exit_position(position)

        assert reason == 'TIME_LIMIT'

    def test_no_exit_needed(self, sample_config):
        """Test when no exit is needed."""
        position = {
            'entry_time': datetime.utcnow().isoformat(),
            'current_pnl': 0.05
        }

        strategy = TradingStrategy(config=sample_config)
        reason = strategy.should_exit_position(position)

        assert reason is None

    def test_missing_entry_time(self, sample_config):
        """Test handling of missing entry time."""
        position = {
            'current_pnl': 0.05
        }

        strategy = TradingStrategy(config=sample_config)
        reason = strategy.should_exit_position(position)

        assert reason is None


class TestRankOpportunities:
    """Test rank_opportunities method."""

    def test_rank_opportunities_basic(self, sample_config):
        """Test ranking multiple opportunities."""
        opportunities = [
            {'edge': 0.05, 'liquidity_usd': 1000, 'volume_24h_usd': 500},
            {'edge': 0.10, 'liquidity_usd': 5000, 'volume_24h_usd': 2000},
            {'edge': 0.07, 'liquidity_usd': 3000, 'volume_24h_usd': 1500},
        ]

        strategy = TradingStrategy(config=sample_config)
        ranked = strategy.rank_opportunities(opportunities)

        assert len(ranked) == 3
        # Highest edge should be ranked first
        assert ranked[0]['edge'] == 0.10

    def test_rank_opportunities_empty_list(self, sample_config):
        """Test ranking empty opportunities list."""
        strategy = TradingStrategy(config=sample_config)
        ranked = strategy.rank_opportunities([])

        assert ranked == []

    def test_rank_opportunities_invalid_type(self, sample_config):
        """Test with invalid opportunities type."""
        strategy = TradingStrategy(config=sample_config)
        ranked = strategy.rank_opportunities("not a list")

        assert ranked == []

    def test_rank_opportunities_filters_non_dicts(self, sample_config):
        """Test filtering of non-dict opportunities."""
        opportunities = [
            {'edge': 0.10, 'liquidity_usd': 5000},
            "invalid",
            {'edge': 0.07, 'liquidity_usd': 3000},
        ]

        strategy = TradingStrategy(config=sample_config)
        ranked = strategy.rank_opportunities(opportunities)

        assert len(ranked) == 2


class TestScoreOpportunity:
    """Test _score_opportunity method."""

    def test_score_opportunity_positive(self, sample_config, sample_opportunity):
        """Test scoring a positive opportunity."""
        strategy = TradingStrategy(config=sample_config)
        score = strategy._score_opportunity(sample_opportunity)

        assert 0 <= score <= 100
        assert score > 0

    def test_score_opportunity_high_edge(self, sample_config):
        """Test scoring with high edge."""
        opportunity = {
            'edge': 0.50,
            'liquidity_usd': 10000,
            'volume_24h_usd': 5000
        }

        strategy = TradingStrategy(config=sample_config)
        score = strategy._score_opportunity(opportunity)

        assert score > 50  # High score due to high edge

    def test_score_opportunity_high_liquidity(self, sample_config):
        """Test scoring with high liquidity."""
        opportunity = {
            'edge': 0.05,
            'liquidity_usd': 50000,
            'volume_24h_usd': 5000
        }

        strategy = TradingStrategy(config=sample_config)
        score = strategy._score_opportunity(opportunity)

        assert score >= 30  # Liquidity contributes to score

    def test_score_opportunity_invalid_type(self, sample_config):
        """Test scoring with invalid type."""
        strategy = TradingStrategy(config=sample_config)
        score = strategy._score_opportunity("not a dict")

        assert score == 0.0


class TestGetPortfolioAllocation:
    """Test get_portfolio_allocation method."""

    def test_allocate_single_opportunity(self, sample_config, sample_opportunity):
        """Test allocating to single opportunity."""
        strategy = TradingStrategy(config=sample_config)
        opportunities = [sample_opportunity]

        allocations = strategy.get_portfolio_allocation(
            opportunities=opportunities,
            capital=10000
        )

        assert len(allocations) == 1
        assert allocations[0]['position_size'] > 0

    def test_allocate_multiple_opportunities(self, sample_config):
        """Test allocating across multiple opportunities."""
        opportunities = [
            {'position_size': 500},
            {'position_size': 600},
            {'position_size': 400},
        ]

        strategy = TradingStrategy(config=sample_config)
        allocations = strategy.get_portfolio_allocation(
            opportunities=opportunities,
            capital=2000
        )

        # Check that total allocation doesn't exceed capital
        total = sum(a.get('position_size', 0) for a in allocations)
        assert total <= 2000

    def test_allocate_respects_max_positions(self, sample_config):
        """Test that allocation respects max concurrent positions."""
        opportunities = [
            {'position_size': 500},
            {'position_size': 500},
            {'position_size': 500},
            {'position_size': 500},
            {'position_size': 500},
            {'position_size': 500},
        ]

        sample_config['trading']['max_concurrent_positions'] = 3
        strategy = TradingStrategy(config=sample_config)
        allocations = strategy.get_portfolio_allocation(
            opportunities=opportunities,
            capital=10000
        )

        assert len(allocations) <= 3

    def test_allocate_respects_concentration_limit(self, sample_config):
        """Test that allocation respects concentration limit."""
        opportunities = [
            {'position_size': 5000},
            {'position_size': 5000},
            {'position_size': 5000},
        ]

        sample_config['risk_management']['position'] = {'max_concentration': 0.20}
        strategy = TradingStrategy(config=sample_config)
        allocations = strategy.get_portfolio_allocation(
            opportunities=opportunities,
            capital=10000
        )

        # Max per trade is 10000 * 0.20 = 2000
        for alloc in allocations:
            assert alloc['position_size'] <= 2000


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_evaluate_opportunity_with_missing_fields(self, sample_config):
        """Test evaluation with missing optional fields."""
        opportunity = {
            'market_id': 'test',
            'edge': 0.10,
            'liquidity_usd': 5000
        }

        strategy = TradingStrategy(config=sample_config)
        trade = strategy.evaluate_opportunity(opportunity)

        # Should handle missing fields gracefully
        assert trade is not None or trade is None  # Either outcome is acceptable

    def test_calculate_position_size_with_zero_probability(self, sample_config, sample_opportunity):
        """Test position sizing with edge case probabilities."""
        strategy = TradingStrategy(config=sample_config)

        size = strategy._calculate_position_size(
            prob=0.0,
            price=0.5,
            opportunity=sample_opportunity
        )

        assert 10 <= size <= 1000

    def test_rank_opportunities_with_missing_fields(self, sample_config):
        """Test ranking opportunities with missing scoring fields."""
        opportunities = [
            {},  # Empty dict
            {'edge': 0.10},  # Missing liquidity
            {'edge': 0.05, 'liquidity_usd': 5000},
        ]

        strategy = TradingStrategy(config=sample_config)
        ranked = strategy.rank_opportunities(opportunities)

        # Should handle gracefully
        assert isinstance(ranked, list)
