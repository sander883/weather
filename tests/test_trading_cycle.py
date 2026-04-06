"""
Test trading cycle to verify list/dict type handling.

This test ensures that the trading cycle properly handles:
1. Opportunities returned as list of dicts
2. Strategy evaluation returning dicts
3. Risk manager checking dicts
4. Trade execution with dicts
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_find_trading_opportunities():
    """Test that find_trading_opportunities returns proper types."""
    from src.main import PolymarketWeatherAgent

    agent = PolymarketWeatherAgent()

    # Test with sample predictions
    test_predictions = {
        'New York': {
            'rain_probability': 0.75,
            'no_rain_probability': 0.25,
            'prediction': 'rain',
            'confidence': 0.75
        }
    }

    opportunities = agent.find_trading_opportunities(test_predictions)

    # Assert it returns a list
    assert isinstance(opportunities, list), f"Expected list, got {type(opportunities)}"

    # Assert each element is a dict
    for opp in opportunities:
        assert isinstance(opp, dict), f"Expected dict, got {type(opp)}"
        assert 'market_id' in opp, "Missing market_id"
        assert 'action' in opp, "Missing action"
        assert 'position_size' in opp, "Missing position_size"

    print(f"✓ find_trading_opportunities test passed ({len(opportunities)} opportunities)")


def test_execute_trades():
    """Test that execute_trades handles dict trades properly."""
    from src.main import PolymarketWeatherAgent

    agent = PolymarketWeatherAgent()

    # Create sample trades
    test_trades = [
        {
            'market_id': '0x123abc',
            'question': 'Will it rain in NYC?',
            'location': 'New York',
            'action': 'BUY_YES',
            'position_size': 500,
            'entry_price': 0.5,
            'implied_probability': 0.7,
            'edge': 0.2
        }
    ]

    # Execute trades (should not raise errors)
    try:
        agent.execute_trades(test_trades)
        print("✓ execute_trades test passed")
    except TypeError as e:
        if "'list' object has no attribute 'get'" in str(e):
            raise AssertionError(f"List/dict error still present: {e}")
        raise


def test_mapper_returns_list_of_dicts():
    """Test that mapper returns list of dicts."""
    from src.market_mapping.mapper import MarketMapper

    mapper = MarketMapper()

    test_predictions = {
        'New York': {'rain_probability': 0.7}
    }

    opportunities = mapper.find_opportunities(test_predictions, edge_threshold=0.05)

    # Assert it's a list
    assert isinstance(opportunities, list), f"Expected list, got {type(opportunities)}"

    # Assert elements are dicts
    for opp in opportunities:
        assert isinstance(opp, dict), f"Expected dict, got {type(opp)}"

    print(f"✓ mapper returns list of dicts test passed")


def test_strategy_returns_dict_or_none():
    """Test that strategy evaluation returns dict or None."""
    from src.market_mapping.mapper import MarketMapper
    from src.trading.strategy import TradingStrategy
    from src.utils.helpers import load_config

    mapper = MarketMapper()
    config = load_config()
    strategy = TradingStrategy(config)

    test_predictions = {
        'New York': {'rain_probability': 0.7}
    }

    opportunities = mapper.find_opportunities(test_predictions, edge_threshold=0.05)

    for opp in opportunities:
        trade = strategy.evaluate_opportunity(opp)

        # Trade should be either None or dict
        assert trade is None or isinstance(trade, dict), \
            f"Expected None or dict, got {type(trade)}"

        if trade is not None:
            assert 'market_id' in trade, "Trade missing market_id"
            assert 'action' in trade, "Trade missing action"

    print("✓ strategy returns dict or None test passed")


def test_risk_manager_with_dict():
    """Test that risk manager accepts dict trades."""
    from src.trading.risk_manager import RiskManager
    from src.utils.helpers import load_config

    config = load_config()
    risk_manager = RiskManager(config)

    test_trade = {
        'market_id': '0x123',
        'action': 'BUY_YES',
        'position_size': 500
    }

    # Should not raise errors
    is_feasible, reason = risk_manager.check_trade_feasibility(test_trade, [])

    assert isinstance(is_feasible, bool), "Expected bool from check_trade_feasibility"
    assert isinstance(reason, str), "Expected str reason"

    print("✓ risk manager with dict test passed")


if __name__ == '__main__':
    import os
    from pathlib import Path

    # Load .env
    env_file = Path(__file__).parent.parent / '.env'
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

    print("Running trading cycle tests...\n")
    test_find_trading_opportunities()
    test_execute_trades()
    test_mapper_returns_list_of_dicts()
    test_strategy_returns_dict_or_none()
    test_risk_manager_with_dict()
    print("\n✓ All tests passed!")
