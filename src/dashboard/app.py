"""Flask dashboard application for bot monitoring."""

from flask import Flask, render_template, jsonify, request
from datetime import datetime
import json
from pathlib import Path

from src.dashboard.tracker import DashboardTracker
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DashboardApp:
    """Flask application for dashboard."""

    def __init__(self, tracker: DashboardTracker, config: dict = None):
        """
        Initialize dashboard app.

        Args:
            tracker: DashboardTracker instance
            config: Configuration dict
        """
        self.tracker = tracker
        self.config = config or {}
        self.app = Flask(__name__)
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Setup Flask routes."""
        @self.app.route('/')
        def index():
            """Dashboard home page."""
            return self._render_dashboard()

        @self.app.route('/api/summary')
        def api_summary():
            """Get dashboard summary."""
            return jsonify(self.tracker.get_summary())

        @self.app.route('/api/performance')
        def api_performance():
            """Get performance metrics."""
            return jsonify(self.tracker.get_performance_metrics())

        @self.app.route('/api/trades')
        def api_trades():
            """Get recent trades."""
            limit = request.args.get('limit', 50, type=int)
            return jsonify(self.tracker.trades[-limit:])

        @self.app.route('/api/trades/<trade_id>')
        def api_trade_detail(trade_id):
            """Get specific trade details."""
            trade = next((t for t in self.tracker.trades if t.get('trade_id') == trade_id), None)
            if trade:
                return jsonify(trade)
            return jsonify({'error': 'Trade not found'}), 404

        @self.app.route('/api/portfolio/history')
        def api_portfolio_history():
            """Get portfolio history."""
            days = request.args.get('days', 7, type=int)
            return jsonify(self.tracker.get_portfolio_history(days))

        @self.app.route('/api/daily-metrics')
        def api_daily_metrics():
            """Get daily metrics."""
            date = request.args.get('date')
            return jsonify(self.tracker.get_daily_metrics(date))

        @self.app.route('/api/alerts')
        def api_alerts():
            """Get recent alerts."""
            limit = request.args.get('limit', 20, type=int)
            return jsonify(self.tracker.alerts[-limit:])

        @self.app.route('/api/health')
        def api_health():
            """Health check endpoint."""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'trades_recorded': len(self.tracker.trades),
                'snapshots_recorded': len(self.tracker.portfolio_snapshots)
            })

        @self.app.route('/api/export')
        def api_export():
            """Export metrics."""
            filepath = self.tracker.export_metrics()
            if filepath:
                return jsonify({'status': 'success', 'path': filepath})
            return jsonify({'status': 'error'}), 500

        @self.app.route('/metrics')
        def metrics():
            """Prometheus metrics endpoint."""
            metrics_text = self._generate_prometheus_metrics()
            return metrics_text, 200, {'Content-Type': 'text/plain; version=0.0.4'}

    def _render_dashboard(self) -> str:
        """Render dashboard HTML."""
        summary = self.tracker.get_summary()
        performance = self.tracker.get_performance_metrics()

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Polymarket Weather Bot Dashboard</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f172a; color: #e2e8f0; }}
                .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
                header {{ background: #1e293b; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
                h1 {{ color: #60a5fa; margin-bottom: 10px; }}
                .timestamp {{ color: #94a3b8; font-size: 14px; }}
                .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 20px; }}
                .card {{ background: #1e293b; padding: 20px; border-radius: 8px; border: 1px solid #334155; }}
                .card h3 {{ color: #60a5fa; margin-bottom: 10px; font-size: 14px; text-transform: uppercase; }}
                .card .value {{ font-size: 32px; font-weight: bold; color: #10b981; margin: 10px 0; }}
                .card .value.negative {{ color: #ef4444; }}
                .card .label {{ color: #94a3b8; font-size: 12px; }}
                .trades {{ background: #1e293b; padding: 20px; border-radius: 8px; border: 1px solid #334155; }}
                .trades h2 {{ color: #60a5fa; margin-bottom: 15px; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #334155; }}
                th {{ background: #0f172a; color: #60a5fa; font-weight: 600; }}
                tr:hover {{ background: #0f172a; }}
                .status-filled {{ color: #10b981; }}
                .status-pending {{ color: #f59e0b; }}
                .status-closed {{ color: #8b5cf6; }}
                .pnl-positive {{ color: #10b981; }}
                .pnl-negative {{ color: #ef4444; }}
                .alerts {{ background: #1e293b; padding: 20px; border-radius: 8px; border: 1px solid #334155; margin-top: 20px; }}
                .alert {{ padding: 10px; margin: 5px 0; border-left: 4px solid; border-radius: 4px; }}
                .alert.info {{ border-color: #60a5fa; background: rgba(96, 165, 250, 0.1); }}
                .alert.warning {{ border-color: #f59e0b; background: rgba(245, 158, 11, 0.1); }}
                .alert.error {{ border-color: #ef4444; background: rgba(239, 68, 68, 0.1); }}
                .chart {{ background: #1e293b; padding: 20px; border-radius: 8px; border: 1px solid #334155; margin-top: 20px; }}
            </style>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        </head>
        <body>
            <div class="container">
                <header>
                    <h1>🚀 Polymarket Weather Trading Bot</h1>
                    <p class="timestamp">Last Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                </header>

                <div class="grid">
                    <div class="card">
                        <h3>Portfolio Value</h3>
                        <div class="value">${summary.get('portfolio', {}).get('current_capital', 0):.2f}</div>
                        <div class="label">Current Capital</div>
                    </div>

                    <div class="card">
                        <h3>Total P&L</h3>
                        <div class="value {'negative' if summary.get('portfolio', {}).get('total_pnl', 0) < 0 else ''}">${summary.get('portfolio', {}).get('total_pnl', 0):.2f}</div>
                        <div class="label">All Time</div>
                    </div>

                    <div class="card">
                        <h3>Win Rate</h3>
                        <div class="value">{performance.get('win_rate', 0) * 100:.1f}%</div>
                        <div class="label">{performance.get('winning_trades', 0)}W / {performance.get('losing_trades', 0)}L</div>
                    </div>

                    <div class="card">
                        <h3>Total Trades</h3>
                        <div class="value">{summary.get('total_trades', 0)}</div>
                        <div class="label">{summary.get('closed_trades', 0)} Closed, {summary.get('open_trades', 0)} Open</div>
                    </div>

                    <div class="card">
                        <h3>Avg P&L</h3>
                        <div class="value {'negative' if performance.get('avg_pnl', 0) < 0 else ''}">${performance.get('avg_pnl', 0):.2f}</div>
                        <div class="label">Per Trade</div>
                    </div>

                    <div class="card">
                        <h3>Profit Factor</h3>
                        <div class="value">{performance.get('profit_factor', 0):.2f}</div>
                        <div class="label">Gains vs Losses</div>
                    </div>
                </div>

                <div class="trades">
                    <h2>Recent Trades</h2>
                    <table>
                        <thead>
                            <tr>
                                <th>Time</th>
                                <th>Market</th>
                                <th>Action</th>
                                <th>Size</th>
                                <th>Entry</th>
                                <th>Exit</th>
                                <th>P&L</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody>
        """

        for trade in summary.get('recent_trades', [])[-10:]:
            time = datetime.fromisoformat(trade['timestamp']).strftime('%H:%M:%S')
            action = trade.get('action', 'N/A')
            size = trade.get('position_size', 0)
            entry = trade.get('entry_price', 'N/A')
            exit_price = trade.get('exit_price', '-')
            pnl = trade.get('pnl', '-')
            status = trade.get('status', 'PENDING')

            pnl_class = 'pnl-positive' if isinstance(pnl, (int, float)) and pnl > 0 else 'pnl-negative'
            pnl_display = f"${pnl:.2f}" if isinstance(pnl, (int, float)) else pnl
            status_class = f'status-{status.lower()}'

            html += f"""
                            <tr>
                                <td>{time}</td>
                                <td>{trade.get('market_id', 'N/A')}</td>
                                <td>{action}</td>
                                <td>{size}</td>
                                <td>${entry}</td>
                                <td>{f'${exit_price}' if exit_price != '-' else '-'}</td>
                                <td class="{pnl_class}">{pnl_display}</td>
                                <td class="{status_class}">{status}</td>
                            </tr>
            """

        html += """
                        </tbody>
                    </table>
                </div>

                <div class="alerts">
                    <h2>Recent Alerts</h2>
        """

        for alert in summary.get('recent_alerts', []):
            severity = alert.get('severity', 'INFO').lower()
            alert_class = f"alert {severity}"
            html += f"""
                    <div class="{alert_class}">
                        <strong>[{alert.get('type', 'SYSTEM')}]</strong> {alert.get('message', '')}
                    </div>
            """

        html += """
                </div>

                <script>
                    // Auto-refresh every 10 seconds
                    setTimeout(() => location.reload(), 10000);

                    // API endpoints available:
                    // GET /api/summary - Dashboard summary
                    // GET /api/performance - Performance metrics
                    // GET /api/trades - Recent trades
                    // GET /api/portfolio/history - Portfolio history
                    // GET /api/daily-metrics - Daily metrics
                    // GET /api/alerts - Recent alerts
                    // GET /api/health - Health status
                    // GET /metrics - Prometheus metrics
                </script>
            </div>
        </body>
        </html>
        """

        return html

    def _generate_prometheus_metrics(self) -> str:
        """Generate Prometheus metrics format."""
        summary = self.tracker.get_summary()
        performance = self.tracker.get_performance_metrics()

        metrics = f"""# HELP bot_portfolio_value Current portfolio value in USD
# TYPE bot_portfolio_value gauge
bot_portfolio_value {summary.get('portfolio', {}).get('current_capital', 0)}

# HELP bot_total_pnl Total P&L in USD
# TYPE bot_total_pnl gauge
bot_total_pnl {summary.get('portfolio', {}).get('total_pnl', 0)}

# HELP bot_total_trades Total number of trades
# TYPE bot_total_trades counter
bot_total_trades {summary.get('total_trades', 0)}

# HELP bot_closed_trades Number of closed trades
# TYPE bot_closed_trades counter
bot_closed_trades {summary.get('closed_trades', 0)}

# HELP bot_open_trades Number of open trades
# TYPE bot_open_trades gauge
bot_open_trades {summary.get('open_trades', 0)}

# HELP bot_win_rate Win rate percentage
# TYPE bot_win_rate gauge
bot_win_rate {performance.get('win_rate', 0)}

# HELP bot_avg_pnl Average P&L per trade
# TYPE bot_avg_pnl gauge
bot_avg_pnl {performance.get('avg_pnl', 0)}

# HELP bot_profit_factor Profit factor
# TYPE bot_profit_factor gauge
bot_profit_factor {performance.get('profit_factor', 0)}

# HELP bot_alerts_total Total number of alerts
# TYPE bot_alerts_total counter
bot_alerts_total {len(self.tracker.alerts)}
"""

        return metrics

    def run(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False) -> None:
        """
        Run Flask app.

        Args:
            host: Host to bind to
            port: Port to bind to
            debug: Enable debug mode
        """
        logger.info(f"Starting dashboard on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug, use_reloader=False)
