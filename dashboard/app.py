"""Flask dashboard for Polymarket Weather Agent."""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from datetime import datetime, timedelta
import json
from pathlib import Path

from src.utils.helpers import load_config
from src.trading.executor import OrderExecutor
from src.trading.risk_manager import RiskManager
from src.market_mapping.mapper import MarketMapper
from src.learning.feedback_loop import LearningLoop

app = Flask(__name__)
CORS(app)

# Load configuration
config = load_config()

# Initialize components
executor = OrderExecutor(config, simulate=True)
risk_manager = RiskManager(config)
mapper = MarketMapper()
learning_loop = LearningLoop(config)


@app.route('/')
def index():
    """Dashboard home page."""
    return render_template('index.html')


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get agent status."""
    portfolio = risk_manager.get_portfolio_stats()
    execution = executor.get_execution_stats()
    learning = learning_loop.get_learning_status()

    return jsonify({
        'timestamp': datetime.utcnow().isoformat(),
        'portfolio': portfolio,
        'execution': execution,
        'learning': learning,
        'status': 'RUNNING' if not risk_manager.circuit_breaker_triggered else 'STOPPED'
    })


@app.route('/api/positions', methods=['GET'])
def get_positions():
    """Get open positions."""
    positions = executor.get_open_positions()

    return jsonify({
        'timestamp': datetime.utcnow().isoformat(),
        'open_positions': positions,
        'count': len(positions)
    })


@app.route('/api/trades', methods=['GET'])
def get_trades():
    """Get trade history."""
    limit = request.args.get('limit', 100, type=int)
    trades = executor.get_execution_history(limit=limit)

    return jsonify({
        'timestamp': datetime.utcnow().isoformat(),
        'trades': trades,
        'count': len(trades)
    })


@app.route('/api/markets', methods=['GET'])
def get_markets():
    """Get market information."""
    markets = mapper.get_active_markets()
    summary = mapper.get_market_summary()

    return jsonify({
        'timestamp': datetime.utcnow().isoformat(),
        'markets': markets,
        'count': len(markets),
        'summary': summary.to_dict('records') if not summary.empty else []
    })


@app.route('/api/risk', methods=['GET'])
def get_risk():
    """Get risk report."""
    report = risk_manager.get_risk_report()

    return jsonify(report)


@app.route('/api/performance', methods=['GET'])
def get_performance():
    """Get performance metrics."""
    execution = executor.get_execution_stats()

    metrics = {
        'total_trades': execution.get('total_trades', 0),
        'closed_trades': execution.get('closed_trades', 0),
        'win_rate': execution.get('win_rate', 0),
        'total_pnl': execution.get('total_pnl', 0),
        'avg_pnl': execution.get('avg_pnl', 0),
        'sharpe_ratio': calculate_sharpe_ratio(executor.get_execution_history()),
        'max_drawdown': risk_manager._calculate_max_drawdown()
    }

    return jsonify({
        'timestamp': datetime.utcnow().isoformat(),
        'metrics': metrics
    })


@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    """Get recent predictions."""
    limit = request.args.get('limit', 50, type=int)

    predictions = learning_loop._load_feedback_data('predictions.jsonl', limit)

    return jsonify({
        'timestamp': datetime.utcnow().isoformat(),
        'predictions': predictions[-limit:],
        'count': len(predictions)
    })


@app.route('/api/learning', methods=['GET'])
def get_learning():
    """Get learning system status."""
    status = learning_loop.get_learning_status()
    evaluation = learning_loop.evaluate_predictions()

    return jsonify({
        'timestamp': datetime.utcnow().isoformat(),
        'status': status,
        'evaluation': evaluation
    })


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get configuration (sanitized)."""
    safe_config = {
        'trading': config.get('trading', {}),
        'risk_management': config.get('risk_management', {}),
        'features': config.get('features', {}),
        'weather': {
            'locations': config.get('weather', {}).get('locations', []),
            'refresh_interval': config.get('weather', {}).get('refresh_interval')
        }
    }

    return jsonify(safe_config)


@app.route('/api/export', methods=['POST'])
def export_data():
    """Export trading data."""
    try:
        export_dir = 'data/exports'
        Path(export_dir).mkdir(parents=True, exist_ok=True)

        # Export trades
        trades = executor.get_execution_history(limit=10000)
        with open(f'{export_dir}/trades.json', 'w') as f:
            json.dump(trades, f, indent=2, default=str)

        # Export learning data
        learning_loop.export_learning_data(export_dir)

        return jsonify({
            'status': 'success',
            'message': f'Data exported to {export_dir}',
            'timestamp': datetime.utcnow().isoformat()
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    })


def calculate_sharpe_ratio(trades):
    """Calculate Sharpe ratio from trades."""
    if not trades:
        return 0.0

    pnls = [t.get('pnl', 0) for t in trades if t.get('pnl') is not None]

    if not pnls:
        return 0.0

    import numpy as np
    pnls = np.array(pnls)

    if pnls.std() == 0:
        return 0.0

    return float((pnls.mean() / pnls.std()) * np.sqrt(252))


if __name__ == '__main__':
    app.run(
        host=config.get('dashboard', {}).get('host', '0.0.0.0'),
        port=config.get('dashboard', {}).get('port', 5000),
        debug=config.get('dashboard', {}).get('debug', False)
    )
